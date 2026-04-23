# ======================== Libraries ========================
import torch
from torch_geometric.nn import GCNConv, VGAE, global_add_pool
import os
import numpy as np

# ======================== Paths ========================
cmap_vgae_weight_file = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\structure-n-seq\plotsfinal\best_cmap_vgae.pt"
af_graph_folder = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\prediction\alpha_cmap_graph_datas"   # <-- AlphaFold graph .pt files
saving_path = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\prediction\af_cmap_embeddings.pt"

# ======================== CMAP VGAE ========================
class CmapEncoder(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv_mu = GCNConv(in_size, out_size)
        self.conv_logstd = GCNConv(in_size, out_size)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VGAE(CmapEncoder(1024, 200)).to(device)
model.load_state_dict(
    torch.load(cmap_vgae_weight_file,
    map_location=device,
    weights_only=True))
model.eval()

# ======================== Safe Pooling ========================
def genPoolEmbedd(data):
    data = data.to(device)

    if data.edge_index.max().item() >= data.num_nodes:
        raise ValueError("Invalid edge_index detected")

    z = model.encode(data.x, data.edge_index)

    pooled = global_add_pool(
        z,
        torch.zeros(z.size(0), dtype=torch.long, device=device)
    )

    return pooled.squeeze().cpu().numpy()

# ======================== MAIN ========================
from tqdm import tqdm

# ======================== MAIN ========================
if __name__ == "__main__":

    embeddings = {}
    skipped = 0

    files = [f for f in os.listdir(af_graph_folder) if f.endswith('.pt')]
    total = len(files)

    print("Total AF graphs:", total)
    print("Using device:", device)

    with torch.no_grad():   # ✅ Move outside loop
        for file in tqdm(files):

            chain_id = file.replace('.pt', '')

            try:
                data = torch.load(
                    os.path.join(af_graph_folder, file),
                    weights_only=False
                )

                emb = genPoolEmbedd(data)

                if emb.shape[0] != 200:
                    skipped += 1
                    continue

                embeddings[chain_id] = emb

            except Exception as e:
                skipped += 1
                continue

    torch.save({
        "embeddings": embeddings,
        "num_samples": len(embeddings),
        "embedding_dim": 200
    }, saving_path)

    print("✅ AlphaFold CMAP Embeddings Saved")
    print("Total processed:", len(embeddings))
    print("Skipped:", skipped)