import pandas as pd
import torch
from torch_geometric.data import Data
import sys
import pickle
from torch_geometric.nn import Node2Vec

# --- INPUT FILE ---
HUMAN_LINKS_FILE = r'D:\abi\abi\pythonProject\Abi 2025\Research\GNN\9606.protein.links.v12.0.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- READ AND FILTER PPI DATA ---
try:
    df = pd.read_csv(HUMAN_LINKS_FILE, sep=" ")

    if 'combined_score' in df.columns:
        print(f"Total links before filtering: {len(df)}")
        df = df[df['combined_score'] >= 700].reset_index(drop=True)
        print(f"Total links after filtering (score >= 700): {len(df)}")
    else:
        print("Warning: 'combined_score' column not found. Processing all links.")

    # Keep only protein1 and protein2
    df = df[['protein1', 'protein2']]

except Exception as e:
    print(f"Error reading or filtering file: {e}")
    sys.exit(1)

# --- MAP PROTEINS TO INDICES ---
all_nodes = pd.unique(df[['protein1', 'protein2']].values.ravel())
node2ind = {node: idx for idx, node in enumerate(all_nodes)}
ind2node = {idx: node for node, idx in node2ind.items()}

df['protein1'] = df['protein1'].map(node2ind)
df['protein2'] = df['protein2'].map(node2ind)

edge_index = torch.tensor([df['protein1'].tolist(), df['protein2'].tolist()], dtype=torch.long)

print(f"Number of unique proteins (nodes): {len(all_nodes)}")
print(f"Number of high-confidence interactions (edges): {edge_index.shape[1]}")

# --- CREATE GRAPH DATA OBJECT ---
data = Data(edge_index=edge_index)

# --- NODE2VEC MODEL ---
embedding_dim = 256
node2vec = Node2Vec(
    edge_index=data.edge_index.to(device),
    embedding_dim=embedding_dim,
    walk_length=10,
    context_size=10,
    walks_per_node=10,
    p=1,
    q=2,
    sparse=True
).to(device)

# Determine number of workers
num_workers = 4 if sys.platform == 'linux' else 0
loader = node2vec.loader(batch_size=64, shuffle=True, num_workers=num_workers)
optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)

# --- TRAINING LOOP ---
def train():
    node2vec.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

for epoch in range(1, 51):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# --- EXTRACT EMBEDDINGS ---
node_embeddings = node2vec().detach().cpu().tolist()

node_embedding_df = pd.DataFrame({
    'protein': [ind2node[idx] for idx in range(len(ind2node))],
    'embedding': node_embeddings
})

# Convert to dictionary for easy lookup
ppi_dict = dict(zip(node_embedding_df['protein'], node_embedding_df['embedding']))

# --- SAVE EMBEDDINGS ---
OUTPUT_FILE = 'node2vec_ppi_embedds_dict.pkl'
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(ppi_dict, f)

print(f"\nSuccessfully generated and saved {len(ppi_dict)} embeddings to {OUTPUT_FILE}")


