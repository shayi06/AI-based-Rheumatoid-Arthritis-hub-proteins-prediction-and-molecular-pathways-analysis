import pandas as pd

# ===== 1. Load Files =====
mapping_file = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN\ppi part\NOde2vec\Ensp_uniID.tsv"
alphafold_file = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\Data_preprocess\alphafold_uniprot_ids.csv"
pdb_file = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN\data_preprocessing\pdb_chain_uniprot.csv"

# Load files
mapping_df = pd.read_csv(mapping_file, sep="\t")
alphafold_df = pd.read_csv(alphafold_file)
pdb_df = pd.read_csv(pdb_file, comment='#',dtype=str)

# ===== 2. Extract UniProt list =====
alphafold_uniprots = alphafold_df.iloc[:, 0].tolist()

# ===== 3. Match UniProt IDs (ENSP Mapping) =====
matched = mapping_df[mapping_df['Entry'].isin(alphafold_uniprots)]
ensp_result = matched[['From', 'Entry']].copy()
ensp_result.columns = ['ENSP_ID', 'UniProt_ID']

# ===== 4. Get PDB + Chain mapping =====
pdb_matched = pdb_df[pdb_df['SP_PRIMARY'].isin(alphafold_uniprots)]
pdb_result = pdb_matched[['PDB', 'CHAIN', 'SP_PRIMARY']].copy()
pdb_result.columns = ['PDB_ID', 'CHAIN_ID', 'UniProt_ID']

# ===== 6. Merge ENSP + PDB_CHAIN info =====
# ===== 6. Merge ENSP + PDB info (keep separate PDB_ID and CHAIN_ID) =====
final_result = pd.merge(
    ensp_result,
    pdb_result[['UniProt_ID', 'PDB_ID', 'CHAIN_ID']],
    on='UniProt_ID',
    how='left'   # keep all ENSP entries even if no PDB mapping
)

# ===== 7. Find Missing UniProt IDs =====
matched_uniprots = set(ensp_result['UniProt_ID'])
all_uniprots = set(alphafold_uniprots)
missing_uniprots = list(all_uniprots - matched_uniprots)
missing_df = pd.DataFrame(missing_uniprots, columns=['Missing_UniProt_ID'])

# ===== 8. Save Files =====
final_result.to_csv("alphafold_ENSP_PDBCHAIN_mapping.csv", index=False)
missing_df.to_csv("missing_uniprot_ids.csv", index=False)

# ===== 9. Print Summary =====
print("Total UniProt IDs given:", len(alphafold_uniprots))
print("Matched ENSP IDs found:", len(ensp_result))
print("Missing UniProt IDs:", len(missing_uniprots))

print("\nSample Final Output:")
print(final_result.head())
