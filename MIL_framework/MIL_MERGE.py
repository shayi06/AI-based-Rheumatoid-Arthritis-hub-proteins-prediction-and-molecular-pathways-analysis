import torch
import pickle
import pandas as pd
from collections import defaultdict

# ============================================================
# 1️⃣ LOAD CHAIN-LEVEL MERGED DATA
# ============================================================

def load_chain_level_dataset(path):

    print("\n🔹 Loading merged chain-level dataset...")

    data = torch.load(path)

    chains = data["chains"]
    embeddings = data["embeddings"]

    print("Total chains:", len(chains))
    print("Embedding shape:", embeddings.shape)

    chain2emb = {chains[i]: embeddings[i] for i in range(len(chains))}

    print("Example chains:", chains[:5])

    return chain2emb


# ============================================================
# 2️⃣ BUILD CHAIN → UNIPROT MAPPING
# ============================================================

def build_chain_to_protein_map(ensp_uni_path, pdb_uniprot_path):

    print("\n🔹 Building chain → UniProt mapping...")

    # ENSP → UniProt
    ensp2uni_df = pd.read_csv(ensp_uni_path, sep='\t')
    ensp2uni_dict = dict(zip(ensp2uni_df['From'], ensp2uni_df['Entry']))

    # UniProt → PDB chains
    pdb_chain_df = pd.read_csv(pdb_uniprot_path, comment='#', low_memory=False)

    pdb_chain_df["PDB"] = pdb_chain_df["PDB"].str.upper().str.strip()
    pdb_chain_df["CHAIN"] = pdb_chain_df["CHAIN"].str.strip()

    chain2uni = {}

    for _, row in pdb_chain_df.iterrows():
        chain_id = f"{row.PDB}-{row.CHAIN}".strip().upper()
        chain2uni[chain_id] = row.SP_PRIMARY

    print("Total chain→protein mappings:", len(chain2uni))
    print("Example mappings:", list(chain2uni.items())[:5])

    return chain2uni


# ============================================================
# 3️⃣ BUILD MIL BAGS
# ============================================================

def build_mil_bags(chain2emb, chain2uni):

    print("\n🔹 Building MIL protein bags...")

    protein_bags = defaultdict(list)

    for chain_id, emb in chain2emb.items():

        protein_id = chain2uni.get(chain_id)

        if protein_id is None:
            continue

        protein_bags[protein_id].append(emb)

    print("Total proteins with at least one chain:",
          len(protein_bags))

    # Convert lists → tensors
    final_bags = {}
    for protein_id, emb_list in protein_bags.items():
        final_bags[protein_id] = torch.stack(emb_list)

    # Debug statistics
    bag_sizes = [v.shape[0] for v in final_bags.values()]

    print("\n🔎 MIL Statistics")
    print("Min chains per protein:", min(bag_sizes))
    print("Max chains per protein:", max(bag_sizes))
    print("Average chains per protein:",
          sum(bag_sizes) / len(bag_sizes))

    example_protein = list(final_bags.keys())[0]
    print("\nExample protein:", example_protein)
    print("Bag shape:", final_bags[example_protein].shape)

    return final_bags


# ============================================================
# 4️⃣ SAVE MIL DATASET
# ============================================================

def save_mil_dataset(protein_bags, save_path):

    print("\n🔹 Saving MIL dataset...")

    protein_ids = list(protein_bags.keys())
    bags = [protein_bags[p] for p in protein_ids]

    torch.save({
        "protein_ids": protein_ids,
        "bags": bags,
        "num_proteins": len(protein_ids)
    }, save_path)

    print("✅ MIL dataset saved")
    print("Total proteins:", len(protein_ids))


# ============================================================
# 5️⃣ MAIN
# ============================================================

def main():

    merged_chain_dataset = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\final_dataset\merged_dataset.pt"
    ensp_uni_file = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\ppi part\NOde2vec\Ensp_uniID.tsv"
    pdb_unip_mapper_file = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\structure-n-seq\pdb_chain_uniprot.csv"
    save_path = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\mil_protein_dataset.pt"

    chain2emb = load_chain_level_dataset(merged_chain_dataset)

    chain2uni = build_chain_to_protein_map(
        ensp_uni_file,
        pdb_unip_mapper_file
    )

    protein_bags = build_mil_bags(chain2emb, chain2uni)

    save_mil_dataset(protein_bags, save_path)


if __name__ == "__main__":
    main()

# Total chains: 99695
# Embedding shape: torch.Size([99695, 456])
# Example chains: ['10GS-A', '10GS-B', '11GS-A', '11GS-B', '121P-A']
#
# 🔹 Building chain → UniProt mapping...
# Total chain→protein mappings: 831050
# Example mappings: [('101M-A', 'P02185'), ('102L-A', 'P00720'), ('102M-A', 'P02185'), ('103L-A', 'P00720'), ('103M-A', 'P02185')]
#
# 🔹 Building MIL protein bags...
# Total proteins with at least one chain: 6349
#
# 🔎 MIL Statistics
# Min chains per protein: 1
# Max chains per protein: 1548
# Average chains per protein: 15.702472830366986
#
# Example protein: P09211
# Bag shape: torch.Size([144, 456])
#
# 🔹 Saving MIL dataset...
# ✅ MIL dataset saved
# Total proteins: 6349
