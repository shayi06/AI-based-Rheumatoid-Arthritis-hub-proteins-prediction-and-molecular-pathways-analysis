import torch
import pickle
import numpy as np
import pandas as pd
import os

# ======================== PATHS ========================
af_cmap_embedding_file = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\prediction\af_cmap_embeddings.pt"
ppi_embedding_dict_path = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\ppi part\NOde2vec\node2vec_ppi_embedds_dict.pkl"
ensp_uni_file = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\ppi part\NOde2vec\Ensp_uniID.tsv"
saving_path = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\prediction\merged_embeddings.pt"


# ======================== LOAD AF CMAP ========================
af_data = torch.load(af_cmap_embedding_file)
af_embeddings = af_data["embeddings"]

print("Sample AF graphs:", list(af_embeddings.keys())[:5])

# ======================== LOAD PPI ========================
with open(ppi_embedding_dict_path, 'rb') as f:
    ppi_dict_raw = pickle.load(f)

ensp_df = pd.read_csv(ensp_uni_file, sep='\t')
ensp2uni = dict(zip(ensp_df['From'], ensp_df['Entry']))

# Convert PPI → UniProt key
uni2ppi = {}

for ensp, emb in ppi_dict_raw.items():
    ensp_clean = ensp.replace("9606.", "")
    uni = ensp2uni.get(ensp_clean)
    if uni:
        uni2ppi[uni.upper()] = np.array(emb)

print("Sample uni2ppi keys (UniProt IDs):", list(uni2ppi.keys())[:5])
# ======================== MERGE ========================
X = []
uniprot_ids = []
chain_ids = []  # NEW
skipped = 0

for uni, cmap_emb in af_embeddings.items():
    uni_upper = uni.split('-')[0].upper()
    chain = uni.split('-')[1] if '-' in uni else 'A'  # default chain 'A' if not present

    if uni_upper not in uni2ppi:
        skipped += 1
        continue

    ppi_emb = uni2ppi[uni_upper]

    if cmap_emb.shape[0] != 200 or ppi_emb.shape[0] != 256:
        skipped += 1
        continue

    merged = np.concatenate([cmap_emb, ppi_emb])
    X.append(merged)
    uniprot_ids.append(uni_upper)
    chain_ids.append(chain)  # save chain info

X = torch.tensor(np.array(X), dtype=torch.float32)
protein_with_chain = [f"{uid}-{chain}" for uid, chain in zip(uniprot_ids, chain_ids)]

torch.save({
    "X": X,
    "uniprot_ids": uniprot_ids,
    "chain_ids": chain_ids,
    "protein_with_chain": protein_with_chain,
    "num_samples": len(uniprot_ids),
    "embedding_dim": 456
}, saving_path)

print("✅ AlphaFold UniProt-wise Merge Done")
print("Total merged:", len(uniprot_ids))
print("Sample merged proteins:", uniprot_ids[:5],chain_ids[:4])
print("Skipped:", skipped)

# Sample AF graphs: ['A0A024R1R8-A', 'A0A024RBG1-A', 'A0A024RCN7-A', 'A0A075B6H5-A', 'A0A075B6H7-A']
# Sample pdb2ppi keys (UniProt IDs): ['P84085', 'Q15027', 'P53621', 'O75154', 'P35606']
# ✅ AlphaFold UniProt-wise Merge Done
# Total merged: 17374
# Skipped: 5576