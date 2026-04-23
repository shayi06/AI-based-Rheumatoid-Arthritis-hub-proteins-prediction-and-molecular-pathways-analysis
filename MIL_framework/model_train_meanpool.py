import os
import torch
import torch.nn as nn
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
output_excel         = os.path.join(dataset_folder, "protein_predictions_mean_pool_cv.xlsx")

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

labeled_proteins = [p for p in protein_ids if protein_labels.get(p)]
protein_bag_dict = {protein_ids[i]: protein_bags[i]
                    for i in range(len(protein_ids))}

print(f"Total labeled proteins: {len(labeled_proteins)}")

# ===================================================================
# MULTI-HOT LABEL MATRIX
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
# MEAN POOLING MODEL
# ===================================================================
class MeanPooling(nn.Module):
    def forward(self, bag):
        return bag.mean(dim=0)

class MeanMILModel(nn.Module):
    def __init__(self, feat_dim=456):
        super().__init__()
        self.pool       = MeanPooling()
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

# ===================================================================
# TRAIN MEAN POOL + EXTRACT VECTORS
# ===================================================================
def train_mean_and_extract(
        protein_bag_dict,
        train_proteins, train_labels,
        all_proteins,
        feat_dim=456, epochs=30,
        disease_name=""):

    model     = MeanMILModel(feat_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3, weight_decay=1e-4)

    label_tensor = torch.tensor(train_labels.reshape(-1, 1),
                                dtype=torch.float32)
    pos        = (label_tensor == 1).sum()
    neg        = (label_tensor == 0).sum()
    pos_weight = (neg / pos.clamp(min=1)).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(epochs):
        idx_perm = torch.randperm(len(train_proteins))
        for i in idx_perm:
            pid   = train_proteins[i]
            bag   = protein_bag_dict[pid].float().to(DEVICE)
            label = label_tensor[i].to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(bag)
            loss      = criterion(logits, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    vectors = {}
    with torch.no_grad():
        for pid in all_proteins:
            bag          = protein_bag_dict[pid].float().to(DEVICE)
            _, z         = model(bag)
            vectors[pid] = z.cpu().numpy()

    return vectors

# ===================================================================
# ML CLASSIFIERS
# MLP = DirectMLP baseline (will use raw 456-dim)
# Others = MIL pipeline (will use 64-dim from MeanMILModel)
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

    # ← DirectMLP baseline: raw 456-dim, no MIL compression
    "MLP": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=500,
        early_stopping=False,
        learning_rate_init=1e-3,
        alpha=1e-3,
        random_state=42),
}

# ===================================================================
# CV LOOP
# ===================================================================
mskf     = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
indices  = list(range(len(labeled_proteins)))
feat_dim = protein_bag_dict[labeled_proteins[0]].shape[1]

print("\n🔹 10-Fold MultilabelStratified CV")
print("   ✅ Mean pooling — biologically justified")
print("   ✅ MLP = DirectMLP baseline on raw 456-dim")
print("   ✅ LR/SVM/RF/GB = MIL pipeline on 64-dim\n")

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

    for d_idx, disease in enumerate(DISEASE_GROUPS):
        print(f"  {disease:<18}: "
              f"train_pos={Y_train[:, d_idx].sum():3d}  "
              f"test_pos={Y_test[:, d_idx].sum():3d}")

    # ---------------------------------------------------------------
    # STEP 1: Train MeanMILModel → extract 64-dim vectors
    # ---------------------------------------------------------------
    print(f"\n  Training mean pool models (fold {fold})...")
    mean_vectors = {}

    for d_idx, disease in enumerate(DISEASE_GROUPS):
        y_train_single = Y_train[:, d_idx]
        vectors = train_mean_and_extract(
            protein_bag_dict,
            train_proteins, y_train_single,
            all_fold_prots,
            feat_dim=feat_dim,
            epochs=30,
            disease_name=disease
        )
        mean_vectors[disease] = vectors
        print(f"    ✅ {disease} mean pool done")

    # ← NEW: compute raw 456-dim mean pooled vectors for DirectMLP
    print(f"    ✅ Computing raw 456-dim vectors for DirectMLP...")
    raw_vectors = {}
    for pid in all_fold_prots:
        bag            = protein_bag_dict[pid].float()
        raw_vectors[pid] = bag.mean(dim=0).numpy()  # 456-dim!

    # ---------------------------------------------------------------
    # STEP 2: Train ML classifiers
    # ---------------------------------------------------------------
    for model_name, clf_template in MODELS.items():
        for d_idx, disease in enumerate(DISEASE_GROUPS):

            # ← KEY CHANGE: MLP uses raw 456-dim, others use 64-dim
            if model_name == "MLP":
                X_train_raw = np.stack([raw_vectors[p]
                              for p in train_proteins])
                X_test_raw  = np.stack([raw_vectors[p]
                              for p in test_proteins])
            else:
                vec_map     = mean_vectors[disease]
                X_train_raw = np.stack([vec_map[p]
                              for p in train_proteins])
                X_test_raw  = np.stack([vec_map[p]
                              for p in test_proteins])

            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test  = scaler.transform(X_test_raw)

            y_train_d = Y_train[:, d_idx]
            y_test_d  = Y_test[:,  d_idx]

            clf = copy.deepcopy(clf_template)

            pos            = y_train_d.sum()
            neg            = len(y_train_d) - pos
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

    print(f"\n  {'Model':<22} {'DIA':>8} {'CV':>8} {'RA':>8} {'OB':>8}")
    print("  " + "─"*54)
    for model_name in MODELS:
        row = f"  {model_name:<22}"
        for disease in DISEASE_GROUPS:
            auc_list = all_fold_metrics[model_name][disease]["auc"]
            auc      = auc_list[-1] if auc_list else float('nan')
            row += f" {auc:>8.4f}" if not np.isnan(auc) \
                   else f" {'NaN':>8}"
        print(row)

# ===================================================================
# FINAL SUMMARY
# ===================================================================
print("\n" + "="*70)
print("FINAL SUMMARY — Mean ± Std AUC (MIL Pipeline vs DirectMLP Baseline)")
print("="*70)
print(f"\n{'Model':<22} {'DIA':>9} {'CV':>9} {'RA':>9} {'OB':>9} {'MEAN':>9}")
print("─"*68)

summary_data = {}
for model_name in MODELS:
    mean_aucs = []
    row       = f"{model_name:<22}"
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

print("\n── MIL Pipeline (64-dim) ──")
for m in ["LogisticRegression", "SVM_RBF", "RandomForest", "GradientBoosting"]:
    print(f"  {m}")
print("\n── DirectMLP Baseline (456-dim, no MIL) ──")
print(f"  MLP")

# Best model per disease
best_model_per_disease = {}
print("\n🏆 Best model per disease:")
for d_idx, disease in enumerate(DISEASE_GROUPS):
    best_name = max(MODELS.keys(),
                    key=lambda m: summary_data[m][d_idx]
                    if not np.isnan(summary_data[m][d_idx]) else 0)
    best_auc  = summary_data[best_name][d_idx]
    best_model_per_disease[disease] = best_name
    print(f"  {disease:<18}: {best_name:<25} AUC={best_auc:.4f}")

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