# ======================== Libraries ========================
import os
import random
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import negative_sampling, remove_self_loops
from torch_geometric.loader import DataLoader
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ======================== Paths ========================
pt_dir = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\structure-n-seq\cmap_graph_datas"
plot_dir = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\structure-n-seq\plotfinal"
SAVE_PATH = os.path.join(plot_dir, 'best_cmap_vgae.pt')
os.makedirs(plot_dir, exist_ok=True)
assert os.path.exists(pt_dir), "❌ CMAP graph directory not found!"

# ======================== Main ========================
def main():
    # ===== Device =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Device:", device)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ===== Reproducibility =====
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # ===== Hyperparameters =====
    num_features = 1024
    mid_channels = 256
    out_channels = 200
    epochs = 10
    lr = 0.001
    batch_size = 64
    num_workers = 4

    # ===== Load File List =====
    file_list = [os.path.join(pt_dir, f) for f in os.listdir(pt_dir) if f.endswith(".pt")]
    random.shuffle(file_list)
    file_list = file_list[:10000]  # LIMIT dataset
    n = len(file_list)
    train_files = file_list[:int(0.8*n)]
    val_files   = file_list[int(0.8*n):int(0.9*n)]
    test_files  = file_list[int(0.9*n):]

    print(f"📊 Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # ===== Graph Loader =====
    MAX_NODES = 5000  # example threshold, adjust to your memory limits

    def load_graphs(file_list):
        graphs = []
        for f in tqdm(file_list, desc="Loading graphs", leave=False, ncols=100):
            try:
                g = torch.load(f, weights_only=False)

                # Skip if x or edge_index is missing
                if g.x is None or g.edge_index is None:
                    continue

                # Skip graphs with C_beta attribute
                if hasattr(g, 'C_beta'):
                    print(f"⚠ Skipped {f}: contains C_beta")
                    continue

                g.num_nodes = g.x.size(0)

                # Skip graphs that are too large
                if g.num_nodes > MAX_NODES:
                    print(f"⚠ Skipped {f}: too large ({g.num_nodes} nodes)")
                    continue

                # Clean edges
                mask = ((g.edge_index[0] >= 0) & (g.edge_index[0] < g.num_nodes) &
                        (g.edge_index[1] >= 0) & (g.edge_index[1] < g.num_nodes))
                g.edge_index = g.edge_index[:, mask]
                g.edge_index, _ = remove_self_loops(g.edge_index)
                g.edge_index = torch.unique(g.edge_index, dim=1)

                if g.edge_index.size(1) == 0:
                    continue

                g.edge_index = g.edge_index.long()
                g.x = g.x.float()
                graphs.append(g)

            except Exception as e:
                print(f"❌ Skipped {f}: {e}", flush=True)

        return graphs

    print("🟢 Preloading graphs into memory...")
    train_graphs = load_graphs(train_files)
    val_graphs = load_graphs(val_files)
    test_graphs = load_graphs(test_files)

    # ===== Encoder & Model =====
    class CMAPEncoder(torch.nn.Module):
        def __init__(self, in_size, mid_size, out_size):
            super().__init__()
            self.conv_mu = GCNConv(in_size, out_size)
            self.conv_logstd = GCNConv(in_size, out_size)

        def forward(self, x, edge_index):
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

    model = VGAE(CMAPEncoder(num_features, mid_channels, out_channels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ===== Training / Evaluation =====
    def train_epoch(graphs):
        model.train()
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        total_loss, steps = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            batch.num_nodes = batch.x.size(0)
            with torch.cuda.amp.autocast(enabled=use_amp):
                z = model.encode(batch.x, batch.edge_index)
                loss = model.recon_loss(z, batch.edge_index) + model.kl_loss()
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            steps += 1
        return total_loss / steps if steps > 0 else np.nan

    def eval_epoch(graphs, return_distributions=False):
        model.eval()
        aucs, aps = [], []
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                batch.num_nodes = batch.x.size(0)
                z = model.encode(batch.x, batch.edge_index)
                neg_edge_index = negative_sampling(batch.edge_index, num_nodes=batch.num_nodes)
                auc, ap = model.test(z, batch.edge_index, neg_edge_index)
                aucs.append(auc)
                aps.append(ap)
        if return_distributions:
            return np.array(aucs), np.array(aps)
        return np.mean(aucs), np.mean(aps)

    # ===== Training Loop =====
    best_val_auc = -1
    train_loss_hist, val_auc_hist = [], []

    for epoch in range(1, epochs+1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        random.shuffle(train_graphs)  # shuffle in-memory
        train_loss = train_epoch(train_graphs)
        val_auc, _ = eval_epoch(val_graphs)
        train_loss_hist.append(train_loss)
        val_auc_hist.append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), SAVE_PATH)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val AUROC: {val_auc:.4f}")

    # ===== Test =====
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    test_aucs, test_aps = eval_epoch(test_graphs, return_distributions=True)
    test_auc_mean, test_ap_mean = np.mean(test_aucs), np.mean(test_aps)
    print(f"\n✅ Final Test AUROC: {test_auc_mean:.4f}")
    print(f"✅ Final Test AP: {test_ap_mean:.4f}")

    # ===== Plots =====
    sns.set(style="whitegrid")

    # Epoch-wise train vs val plot
    plt.figure(figsize=(8,5))
    plt.plot(train_loss_hist, marker='o', label='Train Loss')
    plt.plot(val_auc_hist, marker='s', label='Validation AUROC')
    plt.xlabel("Epoch")
    plt.title("Training Loss & Validation AUROC")
    plt.legend()
    plt.savefig(os.path.join(plot_dir,"train_val_curve.png"), bbox_inches='tight')
    plt.close()

    # Test distribution plots
    combined_df = pd.DataFrame({
        "Metric": ["AUROC"]*len(test_aucs) + ["AP"]*len(test_aps),
        "Score": np.concatenate([test_aucs, test_aps])
    })

    plt.figure(figsize=(8,5))
    sns.violinplot(x="Metric", y="Score", data=combined_df)
    plt.title("Test AUROC & AP Distribution - Violin")
    plt.savefig(os.path.join(plot_dir,"violin_test_metrics.png"), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,5))
    sns.boxplot(x="Metric", y="Score", data=combined_df)
    plt.title("Test AUROC & AP Distribution - Box")
    plt.savefig(os.path.join(plot_dir,"box_test_metrics.png"), bbox_inches='tight')
    plt.close()


# ======================== Entry ========================
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()