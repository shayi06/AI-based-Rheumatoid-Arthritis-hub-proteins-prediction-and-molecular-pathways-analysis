import os
import torch
import torch.nn as nn
import numpy as np
import joblib
import warnings
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ===================================================================
# PATHS
# ===================================================================
dataset_folder       = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\MIL"
labeled_protein_file = os.path.join(dataset_folder, "labeled_sets", "protein_disease_labels.pt")
save_dir             = os.path.join(dataset_folder, "saved_models")
os.makedirs(save_dir, exist_ok=True)

DISEASE_GROUPS = ["DIABETES", "CARDIOVASCULAR", "RHEUMATOID", "OBESITY"]
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ===================================================================
# LOAD DATA
# ===================================================================
print("\n🔹 Loading ALL labeled protein data...")
data           = torch.load(labeled_protein_file, weights_only=False)
protein_ids    = data["protein_ids"]
protein_bags   = data["protein_bags"]
protein_labels = data["protein_labels"]

labeled_proteins = [p for p in protein_ids if protein_labels.get(p)]
protein_bag_dict = {protein_ids[i]: protein_bags[i]
                    for i in range(len(protein_ids))}

print(f"Total labeled proteins: {len(labeled_proteins)}")
print("  ✅ Using ALL proteins for final training (no CV!)")

# ===================================================================
# MULTI-HOT LABEL MATRIX
# ===================================================================
group2idx = {g: i for i, g in enumerate(DISEASE_GROUPS)}
Y = np.zeros((len(labeled_proteins), len(DISEASE_GROUPS)), dtype=int)
for i, pid in enumerate(labeled_proteins):
    for g in protein_labels.get(pid, []):
        if g in group2idx:
            Y[i, group2idx[g]] = 1

print("\n🔹 Label distribution:")
for d_idx, disease in enumerate(DISEASE_GROUPS):
    pos = Y[:, d_idx].sum()
    print(f"  {disease:<18}: {pos} pos / {len(labeled_proteins)-pos} neg")

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
# TRAIN MeanMILModel + EXTRACT 64-dim VECTORS
# ===================================================================
def train_mil_and_extract(protein_bag_dict, all_proteins,
                           train_labels, feat_dim=456,
                           epochs=30, disease_name=""):

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
        idx_perm = torch.randperm(len(all_proteins))
        for i in idx_perm:
            pid   = all_proteins[i]
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

    return model, vectors

# ===================================================================
# TRAIN FINAL RF + SAVE ALL FILES
# ===================================================================
print("\n" + "="*60)
print("TRAINING FINAL MODELS — RandomForest on ALL 472 proteins")
print("="*60)

feat_dim = protein_bag_dict[labeled_proteins[0]].shape[1]

for d_idx, disease in enumerate(DISEASE_GROUPS):
    print(f"\n🔹 Disease: {disease}")

    y_labels = Y[:, d_idx]
    print(f"  Positive: {y_labels.sum()}  "
          f"Negative: {(y_labels==0).sum()}")

    # -------------------------------------------------------
    # STEP 1: Train MeanMILModel on ALL labeled proteins
    # -------------------------------------------------------
    print(f"  Training MeanMILModel (456→64)...")
    mil_model, vectors = train_mil_and_extract(
        protein_bag_dict,
        labeled_proteins,
        y_labels,
        feat_dim=feat_dim,
        epochs=30,
        disease_name=disease
    )

    # -------------------------------------------------------
    # STEP 2: Prepare features
    # -------------------------------------------------------
    X_raw    = np.stack([vectors[p] for p in labeled_proteins])
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # -------------------------------------------------------
    # STEP 3: Train RF on ALL labeled proteins
    # -------------------------------------------------------
    print(f"  Training RandomForest...")
    rf_clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
    rf_clf.fit(X_scaled, y_labels)

    # -------------------------------------------------------
    # STEP 4: Save all 3 files
    # -------------------------------------------------------
    mil_path    = os.path.join(save_dir, f"mean_mil_{disease}.pt")
    rf_path     = os.path.join(save_dir, f"rf_{disease}.pkl")
    scaler_path = os.path.join(save_dir, f"scaler_{disease}.pkl")

    torch.save(mil_model.state_dict(), mil_path)
    joblib.dump(rf_clf,  rf_path)
    joblib.dump(scaler,  scaler_path)

    print(f"  ✅ Saved: mean_mil_{disease}.pt")
    print(f"  ✅ Saved: rf_{disease}.pkl")
    print(f"  ✅ Saved: scaler_{disease}.pkl")

# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "="*60)
print("ALL MODELS SAVED SUCCESSFULLY")
print("="*60)
print(f"\nSaved to: {save_dir}")
print(f"\nFiles per disease:")
for disease in DISEASE_GROUPS:
    print(f"\n  {disease}:")
    print(f"    mean_mil_{disease}.pt  ← MIL compressor (456→64)")
    print(f"    rf_{disease}.pkl       ← RF classifier")
    print(f"    scaler_{disease}.pkl   ← feature scaler")

print(f"\n✅ Ready for prediction on unlabeled proteins!")
