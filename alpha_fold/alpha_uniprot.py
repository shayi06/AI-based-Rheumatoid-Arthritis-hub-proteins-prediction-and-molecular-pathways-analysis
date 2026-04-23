import os
import pandas as pd

# ---------------------- Paths ----------------------
cif_dir = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\Data_preprocess\alpha_cif_files"
out_file = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\Data_preprocess\alphafold_uniprot_ids.csv"

# ---------------------- Scan CIF files ----------------------
uniprot_ids = []
for f in os.listdir(cif_dir):
    if f.endswith('.cif'):
        # Filename example: AF-Q9Y261-F1-model_v4.cif
        parts = f.split('-')
        if len(parts) > 1:
            uniprot_id = parts[1]
            uniprot_ids.append(uniprot_id)

# ---------------------- Remove duplicates ----------------------
uniprot_ids = list(sorted(set(uniprot_ids)))
print(f"Total unique UniProt IDs extracted: {len(uniprot_ids)}")

# ---------------------- Save to CSV ----------------------
df = pd.DataFrame({'UniProt_ID': uniprot_ids})
os.makedirs(os.path.dirname(out_file), exist_ok=True)
df.to_csv(out_file, index=False)
print(f"✅ Saved UniProt IDs to: {out_file}")

