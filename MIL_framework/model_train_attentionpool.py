import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import copy
import warnings
import random

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ===================================================================
# PATHS
# ===================================================================
dataset_folder       = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\MIL"
labeled_protein_file = os.path.join(dataset_folder, "labeled_sets", "protein_disease_labels.pt")
output_excel         = os.path.join(dataset_folder, "protein_predictions_corrected_cv.xlsx")

DISEASE_GROUPS = ["DIABETES", "CARDIOVASCULAR", "RHEUMATOID", "OBESITY"]
num_classes    = len(DISEASE_GROUPS)
group2idx      = {g: i for i, g in enumerate(DISEASE_GROUPS)}
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ===================================================================
# LOAD DATA
# ===================================================================
print("\n🔹 Loading labeled protein data...")
data           = torch.load(labeled_protein_file, weights_only=False)
protein_ids    = data["protein_ids"]
protein_bags   = data["protein_bags"]
protein_labels = data["protein_labels"]

# Keep only labeled proteins (472)
labeled_proteins = [p for p in protein_ids if protein_labels.get(p)]
protein_bag_dict = {protein_ids[i]: protein_bags[i]
                    for i in range(len(protein_ids))}

print(f"Total labeled proteins: {len(labeled_proteins)}")

# ===================================================================
# MULTI-HOT LABEL MATRIX  [472 x 4]
# ===================================================================
Y = np.zeros((len(labeled_proteins), num_classes), dtype=int)
for i, pid in enumerate(labeled_proteins):
    for g in protein_labels.get(pid, []):
        if g in group2idx:
            Y[i, group2idx[g]] = 1

print("\n🔹 Label distribution (all 472):")
for d_idx, disease in enumerate(DISEASE_GROUPS):
    pos = Y[:, d_idx].sum()
    print(f"  {disease:<18}: {pos} pos / {len(labeled_proteins) - pos} neg")

# ===================================================================
# ATTENTION POOLING MODEL
# ===================================================================
class AttentionPooling(nn.Module):
    def __init__(self, feat_dim=456, att_dim=128):
        super().__init__()
        self.att_V = nn.Linear(feat_dim, att_dim)
        self.att_U = nn.Linear(feat_dim, att_dim)
        self.att_w = nn.Linear(att_dim, 1)

    def forward(self, bag):
        A_V = torch.tanh(self.att_V(bag))
        A_U = torch.sigmoid(self.att_U(bag))
        A   = self.att_w(A_V * A_U)
        A   = F.softmax(A, dim=0)
        z   = (A * bag).sum(dim=0)
        return z


class AttentionMILModel(nn.Module):
    def __init__(self, feat_dim=456, att_dim=128):
        super().__init__()
        self.pool       = AttentionPooling(feat_dim, att_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, bag):
        z      = self.pool(bag)
        logits = self.classifier(z)
        return logits, z


def train_attention_and_extract(
        protein_bag_dict,
        train_proteins, train_labels,   # only TRAIN split
        all_proteins,                   # train + test (to extract vectors)
        feat_dim=456, epochs=30,
        disease_name=""):
    """
    ✅ Correct order:
       1. Train attention model on TRAIN proteins only
       2. Extract vectors for ALL proteins (train + test)
          using the trained model → no leakage
    """
    model     = AttentionMILModel(feat_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3, weight_decay=1e-4)

    label_tensor = torch.tensor(train_labels.reshape(-1, 1),
                                dtype=torch.float32)
    pos        = (label_tensor == 1).sum()
    neg        = (label_tensor == 0).sum()
    pos_weight = (neg / pos.clamp(min=1)).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ---- TRAIN on train proteins only ----
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        idx_perm   = torch.randperm(len(train_proteins))

        for i in idx_perm:
            pid   = train_proteins[i]
            bag   = protein_bag_dict[pid].float().to(DEVICE)
            label = label_tensor[i].to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(bag)
            loss = criterion(logits, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

    # ---- EXTRACT vectors for train + test (no grad) ----
    model.eval()
    vectors = {}
    with torch.no_grad():
        for pid in all_proteins:
            bag = protein_bag_dict[pid].float().to(DEVICE)
            _, z = model(bag)
            vectors[pid] = z.cpu().numpy()

    return vectors


# ===================================================================
# ML CLASSIFIERS
# ===================================================================
MODELS = {
    "LogisticRegression": LogisticRegression(
        max_iter=2000, random_state=42,
        C=0.1, class_weight='balanced'),

    "SVM_RBF": SVC(
        kernel='rbf', C=1.0, gamma='scale',
        probability=True, random_state=42,
        class_weight='balanced'),

    "RandomForest": RandomForestClassifier(
        n_estimators=500, max_depth=5,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42),

    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42),

    "MLP": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=500,
        early_stopping=False,       # No val set
        learning_rate_init=1e-3,
        alpha=1e-3,
        random_state=42),
}

# ===================================================================
# MULTILABEL STRATIFIED 10-FOLD CV
# ✅ Balanced across all 4 diseases in every fold
# ✅ No validation set — only train / test
# ===================================================================
mskf    = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
indices = list(range(len(labeled_proteins)))
feat_dim = protein_bag_dict[labeled_proteins[0]].shape[1]

print("\n🔹 10-Fold MultilabelStratified CV")
print("   ✅ Attention trained inside each fold (train only)")
print("   ✅ Balanced disease labels in every fold")
print("   ✅ No validation set\n")

all_fold_metrics = {
    m: {d: {"f1": [], "auc": [], "ap": []}
        for d in DISEASE_GROUPS}
    for m in MODELS}

fold_protein_probs = {
    m: {pid: {} for pid in labeled_proteins}
    for m in MODELS}

for fold, (train_idx, test_idx) in enumerate(
        mskf.split(indices, Y), 1):

    print(f"{'='*65}")
    print(f"FOLD {fold}/10  |  Train: {len(train_idx)}  Test: {len(test_idx)}")

    train_proteins = [labeled_proteins[i] for i in train_idx]
    test_proteins  = [labeled_proteins[i] for i in test_idx]
    all_fold_prots = train_proteins + test_proteins

    Y_train = Y[train_idx]
    Y_test  = Y[test_idx]

    # Print disease balance in this fold
    for d_idx, disease in enumerate(DISEASE_GROUPS):
        print(f"  {disease:<18}: "
              f"train_pos={Y_train[:, d_idx].sum():3d}  "
              f"test_pos={Y_test[:, d_idx].sum():3d}")

    # ---------------------------------------------------------------
    # STEP 1: Train attention per disease (on train only)
    #         Extract vectors for train + test
    # ---------------------------------------------------------------
    print(f"\n  Training attention models (fold {fold})...")
    attention_vectors = {}   # disease -> {pid: vector}

    for d_idx, disease in enumerate(DISEASE_GROUPS):
        y_train_single = Y_train[:, d_idx]

        vectors = train_attention_and_extract(
            protein_bag_dict,
            train_proteins, y_train_single,
            all_fold_prots,
            feat_dim=feat_dim,
            epochs=30,
            disease_name=disease
        )
        attention_vectors[disease] = vectors
        print(f"    ✅ {disease} attention done")

    # ---------------------------------------------------------------
    # STEP 2: Train ML classifiers + evaluate
    # ---------------------------------------------------------------
    for model_name, clf_template in MODELS.items():
        for d_idx, disease in enumerate(DISEASE_GROUPS):

            vec_map = attention_vectors[disease]

            X_train_raw = np.stack([vec_map[p] for p in train_proteins])
            X_test_raw  = np.stack([vec_map[p] for p in test_proteins])

            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test  = scaler.transform(X_test_raw)

            y_train_d = Y_train[:, d_idx]
            y_test_d  = Y_test[:,  d_idx]

            clf = copy.deepcopy(clf_template)

            # Sample weights for GradientBoosting
            pos = y_train_d.sum()
            neg = len(y_train_d) - pos
            weight         = neg / (pos + 1e-8)
            sample_weights = np.where(y_train_d == 1, weight, 1.0)

            if model_name == "GradientBoosting":
                clf.fit(X_train, y_train_d,
                        sample_weight=sample_weights)
            else:
                clf.fit(X_train, y_train_d)

            prob = clf.predict_proba(X_test)[:, 1]
            pred = (prob >= 0.5).astype(int)

            f1  = f1_score(y_test_d, pred, zero_division=0)
            auc = roc_auc_score(y_test_d, prob) \
                  if y_test_d.sum() > 0 else np.nan
            ap  = average_precision_score(y_test_d, prob) \
                  if y_test_d.sum() > 0 else np.nan

            all_fold_metrics[model_name][disease]["f1"].append(f1)
            all_fold_metrics[model_name][disease]["auc"].append(auc)
            all_fold_metrics[model_name][disease]["ap"].append(ap)

            for j, pid in enumerate(test_proteins):
                fold_protein_probs[model_name][pid][disease] = {
                    "prob": round(float(prob[j]), 6),
                    "pred": int(pred[j]),
                    "true": int(y_test_d[j]),
                    "fold": fold
                }

    # Quick AUC print after each fold
    print(f"\n  {'Model':<22} {'DIA':>8} {'CV':>8} {'RA':>8} {'OB':>8}")
    print("  " + "─"*54)
    for model_name in MODELS:
        row = f"  {model_name:<22}"
        for disease in DISEASE_GROUPS:
            auc_list = all_fold_metrics[model_name][disease]["auc"]
            auc      = auc_list[-1] if auc_list else float('nan')
            row += f" {auc:>8.4f}" if not np.isnan(auc) else f" {'NaN':>8}"
        print(row)

# ===================================================================
# FINAL SUMMARY
# ===================================================================
print("\n" + "="*70)
print("FINAL SUMMARY — Mean ± Std AUC across 10 folds")
print("="*70)
print(f"\n{'Model':<22} {'DIA':>9} {'CV':>9} {'RA':>9} {'OB':>9} {'MEAN':>9}")
print("─"*68)

summary_data = {}
for model_name in MODELS:
    mean_aucs = []
    row = f"{model_name:<22}"
    for disease in DISEASE_GROUPS:
        aucs     = [a for a in all_fold_metrics[model_name][disease]["auc"]
                    if not np.isnan(a)]
        mean_auc = np.mean(aucs) if aucs else float('nan')
        std_auc  = np.std(aucs)  if aucs else float('nan')
        mean_aucs.append(mean_auc)
        row += f" {mean_auc:>5.3f}±{std_auc:.2f}"
    overall = np.nanmean(mean_aucs)
    row += f" {overall:>9.4f}"
    print(row)
    summary_data[model_name] = mean_aucs

# Best model per disease
best_model_per_disease = {}
print("\n🏆 Best model per disease:")
for d_idx, disease in enumerate(DISEASE_GROUPS):
    best_name = max(MODELS.keys(),
                    key=lambda m: summary_data[m][d_idx]
                    if not np.isnan(summary_data[m][d_idx]) else 0)
    best_auc = summary_data[best_name][d_idx]
    best_model_per_disease[disease] = best_name
    print(f"  {disease:<18}: {best_name:<25} AUC={best_auc:.4f}")

# Detailed best model stats
print("\n" + "="*70)
print("DETAILED — Best Model per Disease (Mean ± Std)")
print("="*70)
print(f"\n{'Disease':<18} {'Model':<22} {'F1':>7} {'AUC':>7} "
      f"{'StdAUC':>8} {'AP':>7} {'StdAP':>7}")
print("─"*75)
for disease in DISEASE_GROUPS:
    best_name = best_model_per_disease[disease]
    f1s  = all_fold_metrics[best_name][disease]["f1"]
    aucs = [a for a in all_fold_metrics[best_name][disease]["auc"]
            if not np.isnan(a)]
    aps  = [a for a in all_fold_metrics[best_name][disease]["ap"]
            if not np.isnan(a)]
    print(f"{disease:<18} {best_name:<22} "
          f"{np.mean(f1s):>7.4f} "
          f"{np.mean(aucs):>7.4f} "
          f"{np.std(aucs):>8.4f} "
          f"{np.mean(aps):>7.4f} "
          f"{np.std(aps):>7.4f}")

# ===================================================================
# SAVE EXCEL
# ===================================================================
print("\n🔹 Saving Excel...")
rows = []
for pid in labeled_proteins:
    true_labels = protein_labels.get(pid, [])

    row = {
        "Protein"    : pid,
        "True_Labels": ", ".join(true_labels) if true_labels else "None"
    }
    for d_idx, disease in enumerate(DISEASE_GROUPS):
        row[f"True_{disease}"] = 1 if disease in true_labels else 0

    for model_name in MODELS:
        for disease in DISEASE_GROUPS:
            entry = fold_protein_probs[model_name][pid].get(disease, {})
            row[f"{model_name[:4]}_{disease}_Prob"] = entry.get("prob", "")
            row[f"{model_name[:4]}_{disease}_Pred"] = entry.get("pred", "")

    for disease in DISEASE_GROUPS:
        best_name = best_model_per_disease[disease]
        entry     = fold_protein_probs[best_name][pid].get(disease, {})
        row[f"Best_Prob_{disease}"] = entry.get("prob", "")
        row[f"Best_Pred_{disease}"] = entry.get("pred", "")

    rows.append(row)

df_out = pd.DataFrame(rows)
df_out.to_excel(output_excel, index=False)
print(f"✅ Saved: {output_excel}")
print(f"Total rows: {len(df_out)}")