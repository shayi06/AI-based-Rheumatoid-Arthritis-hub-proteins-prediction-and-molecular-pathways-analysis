import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import community.community_louvain as community_louvain

# ================== FILE PATHS ==================
ML_FILE    = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\prediction\predicted_ra_proteins_rf_with_genes.xlsx"
SEED_FILE  = r"D:\abi\abi\pythonProject\Abi 2025\Research\network\seed_retrieving\processed_protein.xlsx"
LINKS_FILE = r"D:\abi\abi\pythonProject\Abi 2025\Research\network\9606.protein.links.v12.0.txt"
INFO_FILE  = r"D:\abi\abi\pythonProject\Abi 2025\Research\network\9606.protein.info.v12.0.txt"
module_path = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\DEG analysis\modules"

# ================== 1. LOAD FILES ==================
print("🔹 STEP 1: Loading files...")

ml_df = pd.read_excel(ML_FILE)
seed_data = pd.read_excel(SEED_FILE)

ml_genes = set(ml_df["Gene_Symbol"].dropna())
seeds    = set(seed_data["preferredName"].dropna())

# Separate novel from ML
novel_set = ml_genes - seeds

# FINAL protein set (IMPORTANT)
all_proteins = novel_set.union(seeds)

# ================== DEBUG COUNTS ==================
print("\n📊 PROTEIN COUNT SUMMARY")
print("=" * 40)
print(f"Total ML proteins        : {len(ml_genes)}")
print(f"Known proteins (seed)    : {len(seeds)}")
print(f"Overlap (ML ∩ Known)     : {len(ml_genes.intersection(seeds))}")
print(f"Novel proteins           : {len(novel_set)}")
print(f"✅ FINAL TOTAL USED       : {len(all_proteins)} (should be 350)")
print("=" * 40)

# ================== 2. LOAD STRING DATA ==================
print("\n🔹 STEP 2: Loading STRING PPI data...")

links = pd.read_table(LINKS_FILE, sep=" ")
info  = pd.read_table(INFO_FILE)

id2gene = dict(zip(info["#string_protein_id"], info["preferred_name"]))

print(f"Total STRING proteins mapped: {len(id2gene)}")

links = links[links["combined_score"] >= 700]
print(f"High-confidence interactions: {len(links)}")

# ================== 3. BUILD SUBNETWORK ==================
print("\n🔹 STEP 3: Building subnetwork...")

G = nx.Graph()
added = 0
skipped = 0

for _, row in links.iterrows():
    g1 = id2gene.get(row["protein1"])
    g2 = id2gene.get(row["protein2"])

    if g1 is None or g2 is None:
        skipped += 1
        continue

    if g1 in all_proteins and g2 in all_proteins:
        G.add_edge(g1, g2, weight=row["combined_score"] / 1000)
        added += 1

print(f"Edges added              : {added}")
print(f"Skipped (mapping issues) : {skipped}")
print(f"Nodes in graph           : {G.number_of_nodes()}")
print(f"Edges in graph           : {G.number_of_edges()}")

# ================== COVERAGE CHECK ==================
proteins_in_graph = set(G.nodes())

print("\n🔍 NETWORK COVERAGE")
print("=" * 40)
print(f"Input proteins           : {len(all_proteins)}")
print(f"Proteins in network      : {len(proteins_in_graph)}")
print(f"Lost proteins            : {len(all_proteins - proteins_in_graph)}")
print("=" * 40)

# ================== 4. CONNECTED COMPONENT ==================
print("\n🔹 STEP 4: Extracting largest connected component...")

if G.number_of_nodes() == 0:
    raise ValueError("❌ Graph is empty — check gene name mapping!")

if not nx.is_connected(G):
    largest = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest).copy()

print(f"Nodes after CC filter : {G.number_of_nodes()}")
print(f"Edges after CC filter : {G.number_of_edges()}")

# ================== 5. LOUVAIN COMMUNITY ==================
print("\n🔹 STEP 5: Running Louvain clustering...")

partitions = community_louvain.best_partition(G, random_state=42)

modules = defaultdict(list)
for node, mod in partitions.items():
    modules[mod].append(node)

print(f"Total modules detected: {len(modules)}")

for mod_id, members in sorted(modules.items()):
    print(f"  Module {mod_id}: {len(members)} proteins")

# Save modules
with open("ml_all_partitions.txt", "w") as f:
    for mod_id, members in modules.items():
        f.write(f"Module {mod_id} ({len(members)} proteins):\n")
        f.write(", ".join(members) + "\n\n")

print("✅ Modules saved!")

# ================== 6. Z-SCORE (CORRECTED) ==================
print("\n🔹 STEP 6: Calculating Z-scores (within-module)...")

z_scores = {}
intra_deg = {}

for mod_id, members in modules.items():

    interactions = []

    # Calculate within-module degree (k_in)
    for node in members:
        k_in = sum(1 for nb in G.neighbors(node) if partitions[nb] == mod_id)
        intra_deg[node] = k_in
        interactions.append(k_in)

    interactions = np.array(interactions)
    mean_k = interactions.mean()
    std_k = interactions.std(ddof=1)

    print(f"Module {mod_id}: mean={mean_k:.2f}, std={std_k:.2f}")

    # Compute Z-score
    for node in members:
        if std_k > 0:
            z = (intra_deg[node] - mean_k) / std_k
        else:
            z = 0
        z_scores[node] = z

# ================== 7. HUB DETECTION ==================
print("\n🔹 STEP 7: Identifying hubs...")

hubs_list = []

for node, z in z_scores.items():
    if z >= 1.5:
        neighbors = list(G.neighbors(node))
        degree = len(neighbors)

        neighbor_mod_counts = defaultdict(int)
        for nb in neighbors:
            neighbor_mod_counts[partitions[nb]] += 1

        pc = 1 - sum((cnt / degree) ** 2 for cnt in neighbor_mod_counts.values()) if degree > 0 else 0

        hub_type = "Connector Hub" if pc > 0.5 else "Module Hub"

        hubs_list.append({
            "Gene": node,
            "Module": partitions[node],
            "Z-score": round(z, 4),
            "PC": round(pc, 4),
            "Hub Type": hub_type,
            "Protein Type": "Known" if node in seeds else "Novel"
        })

hub_df = pd.DataFrame(hubs_list).drop_duplicates(subset=["Gene"])

print(f"Total hubs found: {len(hub_df)}")

# ================== 8. SUMMARY ==================
print("\n" + "=" * 60)
print("FINAL HUB ANALYSIS")
print("=" * 60)

novel_hubs = hub_df[hub_df["Protein Type"] == "Novel"]
known_hubs = hub_df[hub_df["Protein Type"] == "Known"]

print(f"Proteins in graph       : {G.number_of_nodes()}")
print(f"Total modules           : {len(modules)}")
print(f"Total hubs              : {len(hub_df)}")
print(f"Connector hubs          : {len(hub_df[hub_df['Hub Type'] == 'Connector Hub'])}")
print(f"Module hubs             : {len(hub_df[hub_df['Hub Type'] == 'Module Hub'])}")
print(f"Novel hubs              : {len(novel_hubs)}")
print(f"Known hubs              : {len(known_hubs)}")

print("\n🔝 Top 15 hubs:")
print(hub_df.sort_values("Z-score", ascending=False).head(15))

# ================== 9. SAVE ==================
import os

print("\n🔹 STEP 9: Saving outputs...")

# ✅ Make sure the folder exists
os.makedirs(module_path, exist_ok=True)

# Save all hubs together
hub_df.to_excel(os.path.join(module_path, "ml_all_hub_genes.xlsx"), index=False)

# ✅ Save ALL proteins in each module separately
for mod_id, members in sorted(modules.items()):
    mod_rows = []
    for protein in members:
        mod_rows.append({
            "Gene": protein,
            "Module": mod_id,
            "Z-score": round(z_scores.get(protein, 0), 4),
            "Intra_Degree": intra_deg.get(protein, 0),
            "Protein Type": "Known" if protein in seeds else "Novel",
            "Is_Hub": "Yes" if protein in hub_df["Gene"].values else "No"
        })

    mod_df = pd.DataFrame(mod_rows)
    filename = os.path.join(module_path, f"ml_module_{mod_id}_proteins.xlsx")
    mod_df.to_excel(filename, index=False)
    print(f"✅ Saved: {filename}  ({len(mod_df)} proteins)")

nx.write_gml(G, os.path.join(module_path, "ml_all_subnetwork.gml"))

module_map = []

for mod_id, genes in modules.items():
    for g in genes:
        module_map.append({"Gene": g, "Module": mod_id})

pd.DataFrame(module_map).to_excel("cytoscape_modules.xlsx", index=False)

print("✅ Saved: ml_all_hub_genes.xlsx")
print("✅ Saved: ml_all_subnetwork.gml")
print("\n🎉 DONE — PROTEIN NETWORK ANALYSIS COMPLETE!")


# Input proteins           : 350
# Proteins in network      : 317
# Lost proteins            : 33
# Nodes after CC filter : 315 means >0.7
# Total modules detected: 6
#   Module 0: 73 proteins
#   Module 1: 59 proteins
#   Module 2: 5 proteins
#   Module 3: 4 proteins
#   Module 4: 99 proteins
#   Module 5: 75 proteins

# Proteins in graph       : 315
# Total modules           : 6
# Total hubs              : 26
# Connector hubs          : 18
# Module hubs             : 8
# Novel hubs              : 9
# Known hubs              : 17


