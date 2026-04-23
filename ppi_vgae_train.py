import pandas as pd
import numpy as np
import pickle
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, VGAE
from torch_geometric.utils import train_test_split_edges, negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Device and Seeds
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
np.random.seed(42)

# ----------------------------
# Paths
# ----------------------------
human_links_file = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN\9606.protein.links.v12.0.txt"
node2vec_embeddings_file = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN\ppi part\NOde2vec\node2vec_ppi_embedds_dict.pkl"
model_save_path = r'human_ppi_vgae_best.pt'

# ----------------------------
# Load Node2Vec embeddings
# ----------------------------
with open(node2vec_embeddings_file, 'rb') as f:
    node2vec_embeddings = pickle.load(f)

# ----------------------------
# Data Object Generation
# ----------------------------
def gen_data_obj(filepath, node2vec_dict):
    df = pd.read_csv(filepath, sep=' ')
    col1, col2 = df['protein1'].tolist(), df['protein2'].tolist()
    unique_nodes = list(dict.fromkeys(col1 + col2))
    node2idx = {node: idx for idx, node in enumerate(unique_nodes)}
    idx2node = {idx: node for node, idx in node2idx.items()}

    # Map proteins to indices
    df['protein1'] = df['protein1'].map(node2idx)
    df['protein2'] = df['protein2'].map(node2idx)

    # Build undirected edge_index
    src, dst = df['protein1'].tolist(), df['protein2'].tolist()
    src, dst = src + dst, dst + src
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Filter nodes that have embeddings
    nodes_with_emb = [n for n in unique_nodes if n in node2vec_dict]
    filtered_node2idx = {n: i for i, n in enumerate(nodes_with_emb)}

    filtered_edges = []
    for s, d in zip(edge_index[0], edge_index[1]):
        node_s = idx2node[int(s)]
        node_d = idx2node[int(d)]
        if node_s in filtered_node2idx and node_d in filtered_node2idx:
            filtered_edges.append((filtered_node2idx[node_s], filtered_node2idx[node_d]))

    if not filtered_edges:
        raise ValueError("No edges remain after filtering by embeddings.")

    edge_index_filtered = torch.tensor(list(zip(*filtered_edges)), dtype=torch.long)
    node_features = torch.tensor([node2vec_dict[n] for n in nodes_with_emb], dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index_filtered)
    print(f"Data object created: {data.num_nodes} nodes, {data.num_edges // 2} edges")
    return data

# Build data
human_data = gen_data_obj(human_links_file, node2vec_embeddings)

# ----------------------------
# GraphSAGE Encoder
# ----------------------------
class PPIEncoder(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(PPIEncoder, self).__init__()
        self.conv_mu = SAGEConv(in_size, out_size)
        self.conv_logstd = SAGEConv(in_size, out_size)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# ----------------------------
# Train & Test Functions
# ----------------------------
def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    x = data.x.to(device)
    train_pos = data.train_pos_edge_index.to(device)

    z = model.encode(x, train_pos)
    loss = model.recon_loss(z, train_pos)
    loss += (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

def test(model, data, pos_edge_index=None, neg_edge_index=None):
    model.eval()

    if pos_edge_index is None:
        pos_edge_index = data.pos_edge_index
    if neg_edge_index is None:
        neg_edge_index = negative_sampling(edge_index=data.edge_index, num_nodes=data.num_nodes)

    pos_edge_index = pos_edge_index.to(device)
    neg_edge_index = neg_edge_index.to(device)

    with torch.no_grad():
        z = model.encode(data.x.to(device), data.train_pos_edge_index.to(device))

    pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1).cpu().numpy()
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1).cpu().numpy()

    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([pos_scores, neg_scores])

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return auc, ap

# ----------------------------
# Split edges
# ----------------------------
data_split = train_test_split_edges(human_data, val_ratio=0.05, test_ratio=0.10)

# ----------------------------
# Initialize model
# ----------------------------
model = VGAE(PPIEncoder(in_size=256, out_size=200)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# Training Loop
# ----------------------------
epochs = 200
train_losses = []
val_aucs, val_aps = [], []

for epoch in range(1, epochs + 1):
    tr_loss = train(model, optimizer, data_split)
    train_losses.append(tr_loss)

    val_auc, val_ap = test(model, data_split, pos_edge_index=data_split.val_pos_edge_index,
                           neg_edge_index=getattr(data_split, 'val_neg_edge_index', None))
    val_aucs.append(val_auc)
    val_aps.append(val_ap)

    print(f'Epoch {epoch}: Train Loss {tr_loss:.4f}, Val AUC {val_auc:.4f}, Val AP {val_ap:.4f}')

# ----------------------------
# Save final model
# ----------------------------
torch.save(model.state_dict(), model_save_path)

# ----------------------------
# Final Test Evaluation
# ----------------------------
test_auc, test_ap = test(model, data_split, pos_edge_index=data_split.test_pos_edge_index,
                         neg_edge_index=getattr(data_split, 'test_neg_edge_index', None))
print(f"Final Test AUC: {test_auc:.4f}, AP: {test_ap:.4f}")

# ----------------------------
# Plot Learning Curves
# ----------------------------
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.savefig("training_loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), val_aucs, label="Validation AUROC")
plt.plot(range(1, epochs+1), val_aps, label="Validation AP")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation AUROC & AP Curve")
plt.legend()
plt.savefig("validation_auroc_ap_curve.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# Data object created: 16201 nodes, 11956410 edges
# Epoch 1: Train Loss 13.2995, Val AUC 0.7231, Val AP 0.7503
# Epoch 200: Train Loss 3.0064, Val AUC 0.8209, Val AP 0.8178
# Final Test AUC: 0.8205, AP: 0.8171