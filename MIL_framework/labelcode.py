import os
import torch
import pandas as pd
from collections import defaultdict

# ===== FILE PATHS =====
disease_file = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\processed_disease_cleaned.xlsx"
mil_dataset_file = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\MIL\mil_protein_dataset.pt"
saving_dir = os.path.join(os.path.dirname(mil_dataset_file), "labeled_sets")
os.makedirs(saving_dir, exist_ok=True)

# ===== DISEASE GROUPS =====
DISEASE_GROUPS = {
    "DIABETES": ["Diabetes Mellitus,Insulin-Dependent", "Diabetes Mellitus",
                 "Diabetes Mellitus,Non-Insulin-Dependent", "Gestational Diabetes", "Diabetes Insipidus"],
    "CARDIOVASCULAR": ["Heart Diseases", "Coronary heart disease", "Congenital heart disease"],
    "RHEUMATOID": ["Rheumatoid Arthritis,Systemic Juvenile", "Rheumatoid Arthritis"],
    "OBESITY": ["Obesity", "Morbid obesity", "Pediatric Obesity"]
}

# ===== LOAD MIL DATASET =====
print("🔹 Loading MIL dataset...")
mil_data = torch.load(mil_dataset_file)
protein_ids = mil_data["protein_ids"]
protein_bags = mil_data["bags"]
print("Total proteins:", len(protein_ids))

# ===== LOAD DISEASE FILE =====
df = pd.read_excel(disease_file)
uni_labels = {}  # UniProt -> disease group(s)
for _, row in df.iterrows():
    uni_id = str(row['Uniprot accession']).upper().strip()
    disease_name = str(row['Disease']).strip()
    if uni_id not in uni_labels:
        uni_labels[uni_id] = []
    for group, keywords in DISEASE_GROUPS.items():
        if any(k.lower() in disease_name.lower() for k in keywords):
            uni_labels[uni_id].append(group)

# ===== LABEL PROTEINS =====
protein_labels = {}  # protein_id (UniProt) -> disease group(s)
unlabeled_proteins = []

for prot_id in protein_ids:
    labels = uni_labels.get(prot_id, [])
    if not labels:
        unlabeled_proteins.append(prot_id)
    protein_labels[prot_id] = labels

# ===== PRINT SUMMARY =====
print("\n🔹 Summary of protein labels")
all_groups = defaultdict(int)
for labels in protein_labels.values():
    for g in labels:
        all_groups[g] += 1
for g, count in all_groups.items():
    print(f"{g}: {count} proteins")

print(f"Proteins with no disease label: {len(unlabeled_proteins)}")
print(f"Proteins with at least one disease label: {len(protein_ids) - len(unlabeled_proteins)}")

# ===== SAVE =====
save_path = os.path.join(saving_dir, "protein_disease_labels.pt")
torch.save({
    "protein_ids": protein_ids,
    "protein_labels": protein_labels,
    "protein_bags": protein_bags
}, save_path)
print("\n✅ Labeled protein bags saved at:", save_path)

# Total proteins: 6349

# 🔹 Summary of protein labels
# RHEUMATOID: 133 proteins
# DIABETES: 150 proteins
# CARDIOVASCULAR: 113 proteins
# OBESITY: 189 proteins
# Proteins with no disease label: 5877
# Proteins with at least one disease label: 472