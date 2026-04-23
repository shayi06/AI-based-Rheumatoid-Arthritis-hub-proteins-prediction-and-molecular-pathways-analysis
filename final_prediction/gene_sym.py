import requests
import pandas as pd
import time

# ===== YOUR UNIPROT IDs =====
#excel_path = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\prediction\predicted_ra_proteins_rf_novel_only.xlsx"
#excel_path = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\prediction\predicted_ra_proteins_rf_known_recovered.xlsx"
excel_path = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\prediction\predicted_ra_proteins_rf.xlsx"
df = pd.read_excel(excel_path)
uniprot_ids = df["UniProt_ID"].dropna().tolist()

# ===== FETCH GENE SYMBOLS FROM UNIPROT API =====
def get_gene_symbol(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Extract gene symbol
            genes = data.get("genes", [])
            if genes:
                gene_name = genes[0].get("geneName", {}).get("value", "")
                return gene_name
        return "NOT_FOUND"
    except Exception as e:
        return f"ERROR: {e}"

# ===== RUN =====
print(f"🔹 Fetching gene symbols for {len(uniprot_ids)} proteins...")
print("(This may take a moment due to API calls)\n")

results = []
for i, uid in enumerate(uniprot_ids):
    gene = get_gene_symbol(uid)
    results.append({
        "UniProt_ID"  : uid,
        "Gene_Symbol" : gene
    })
    print(f"  [{i+1:>4}/{len(uniprot_ids)}]  {uid:<12} → {gene}")
    time.sleep(0.3)   # polite delay to avoid API rate limit

# ===== MERGE WITH ORIGINAL EXCEL =====
df_genes = pd.DataFrame(results)
df_merged = df.merge(df_genes, on="UniProt_ID", how="left")

# ===== SAVE =====
save_path = excel_path.replace(".xlsx", "_with_genes.xlsx")
df_merged.to_excel(save_path, index=False)
print(f"\n✅ Saved: {save_path}")
print(f"Total proteins processed: {len(results)}")

# ===== QUICK CHECK =====
not_found = df_genes[df_genes["Gene_Symbol"] == "NOT_FOUND"]
print(f"Not found: {len(not_found)}")
if len(not_found) > 0:
    print(not_found["UniProt_ID"].tolist())