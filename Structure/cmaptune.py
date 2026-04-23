# ======================== Libraries ========================
import os
import random
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch_geometric
from tqdm import trange, tqdm
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":

    # ======================== Paths ========================
    # pre_dir = "/home/hpc_users/pasanfernando/USBDisk/Ailakshini/cmap_graph_datas"
    # save_dir = "/home/hpc_users/pasanfernando/USBDisk/Ailakshini/plots"

    pre_dir = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\structure-n-seq\cmap_graph_datas"
    save_dir = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\structure-n-seq\plotfinal"
    os.makedirs(save_dir, exist_ok=True)

    # ======================== Device ========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Device:", device)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ======================== Hyperparameters ========================
    num_features = 1024
    mid_channels = 256
    out_channels = 200
    epochs = 30  # fixed epochs for better visualization
    learning_rates = [0.1, 0.01, 0.001]
    batch_size = 64
    num_workers = 4

    # ======================== Load Graphs ========================
    def load_graphs(file_list):
        graphs = []
        for f in tqdm(file_list, desc="Loading graphs", leave=False, unit="graph", ncols=100):
            try:
                g = torch.load(f, weights_only=False)
                if g.x is None or g.edge_index is None: continue
                g.num_nodes = g.x.size(0)
                mask = ((g.edge_index[0] >= 0) & (g.edge_index[0] < g.num_nodes) &
                        (g.edge_index[1] >= 0) & (g.edge_index[1] < g.num_nodes))
                g.edge_index = g.edge_index[:, mask]
                g.edge_index, _ = torch_geometric.utils.remove_self_loops(g.edge_index)
                g.edge_index = torch.unique(g.edge_index, dim=1)
                if g.edge_index.size(1) == 0: continue
                g.edge_index = g.edge_index.long()
                g.x = g.x.float()
                graphs.append(g)
            except Exception as e:
                print(f"❌ Skipped {f}: {e}", flush=True)
        return graphs

    file_list = [os.path.join(pre_dir, f) for f in os.listdir(pre_dir) if f.endswith(".pt")]
    random.seed(42)
    random.shuffle(file_list)

    file_list = file_list[:50000]  # LIMIT dataset
    n = len(file_list)
    train_files = file_list[:int(0.8*n)]
    val_files   = file_list[int(0.8*n):int(0.9*n)]
    test_files  = file_list[int(0.9*n):]

    print(f"📊 Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    train_graphs = load_graphs(train_files)
    val_graphs   = load_graphs(val_files)
    test_graphs  = load_graphs(test_files)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ======================== Encoders ========================
    class CMAPEncoder1(torch.nn.Module):
        def __init__(self, in_size, mid_size, out_size):
            super().__init__()
            self.conv_mu = GCNConv(in_size, out_size)
            self.conv_logstd = GCNConv(in_size, out_size)
        def forward(self, x, edge_index):
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

    class CMAPEncoder2(torch.nn.Module):
        def __init__(self, in_size, mid_size, out_size):
            super().__init__()
            self.conv1 = GCNConv(in_size, mid_size)
            self.conv_mu = GCNConv(mid_size, out_size)
            self.conv_logstd = GCNConv(mid_size, out_size)
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

    class CMAPEncoder3(torch.nn.Module):
        def __init__(self, in_size, mid_size, out_size):
            super().__init__()
            self.conv1 = GCNConv(in_size, mid_size)
            self.conv2 = GCNConv(mid_size, mid_size)
            self.conv_mu = GCNConv(mid_size, out_size)
            self.conv_logstd = GCNConv(mid_size, out_size)
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

    # ======================== Train/Test Functions ========================
    def train_epoch(loader):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                z = model.encode(batch.x, batch.edge_index)
                loss = model.recon_loss(z, batch.edge_index) + model.kl_loss()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            steps += 1
        return total_loss / steps if steps > 0 else None

    def evaluate(loader):
        model.eval()
        aucs, aps = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                if batch.edge_index.size(1) == 0: continue

                neg_edge_index = negative_sampling(batch.edge_index, num_nodes=batch.num_nodes)
                z = model.encode(batch.x, batch.edge_index)
                auc, ap = model.test(z, batch.edge_index, neg_edge_index)
                aucs.append(auc)
                aps.append(ap)
        return aucs, aps

    # ======================== Hyperparameter Tuning & Metrics Logging ========================
    main_auc_df = pd.DataFrame()
    main_ap_df = pd.DataFrame()

    # Create Excel writer BEFORE starting experiments
    excel_path = os.path.join(save_dir, "all_experiment_metrics.xlsx")
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')

    for lr in learning_rates:
        for i, Encoder in enumerate([CMAPEncoder1, CMAPEncoder2, CMAPEncoder3], 1):
            print(f"\n🔹 Encoder {i} | LR {lr}")

            # Initialize model & optimizer
            model = VGAE(Encoder(num_features, mid_channels, out_channels)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # record metrics per epoch
            epoch_metrics = []

            for epoch in trange(epochs, desc=f"Training E{i} LR{lr}"):
                train_loss = train_epoch(train_loader)
                val_aucs, val_aps = evaluate(val_loader)
                test_aucs, test_aps = evaluate(test_loader)

                # Compute mean metrics per epoch
                mean_val_auc = np.mean(val_aucs) if len(val_aucs) > 0 else np.nan
                mean_val_ap  = np.mean(val_aps)  if len(val_aps) > 0 else np.nan
                mean_test_auc = np.mean(test_aucs) if len(test_aucs) > 0 else np.nan
                mean_test_ap  = np.mean(test_aps)  if len(test_aps) > 0 else np.nan

                epoch_metrics.append({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_auc": mean_val_auc,
                    "val_ap": mean_val_ap,
                    "test_auc": mean_test_auc,
                    "test_ap": mean_test_ap
                })

            # Save metrics to Excel sheet for this encoder + LR
            metrics_df = pd.DataFrame(epoch_metrics)
            sheet_name = f"E{i}_LR{lr}"
            metrics_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"📄 Added sheet: {sheet_name}")

            # Save best model based on val AUC
            best_epoch = max(epoch_metrics, key=lambda x: x["val_auc"])
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"best_E{i}_lr{lr}_epoch{best_epoch['epoch']}.pt")
            )
            print(f"🏆 Best Val AUC: {best_epoch['val_auc']:.4f} at epoch {best_epoch['epoch']}")

            # ======================== Prepare DataFrame for Plotting ========================
            for m in epoch_metrics:
                tag = f"GCN layers-{i}, lr-{lr}"
                main_auc_df = pd.concat([main_auc_df, pd.DataFrame({
                    "AUROC": [m["val_auc"], m["test_auc"]],
                    "Distribution": ["Validation", "Test"],
                    "Parameters": tag,
                    "Epoch": m["epoch"]
                })], ignore_index=True)
                main_ap_df = pd.concat([main_ap_df, pd.DataFrame({
                    "AP": [m["val_ap"], m["test_ap"]],
                    "Distribution": ["Validation", "Test"],
                    "Parameters": tag,
                    "Epoch": m["epoch"]
                })], ignore_index=True)

    # CLOSE Excel writer after all experiments
    writer.close()
    print(f"✅ Saved full Excel file → {excel_path}")

    # ======================== Plots ========================
    sns.set(style="whitegrid")
    sns.catplot(x="Distribution", y="AUROC", hue="Parameters", data=main_auc_df, kind="violin", height=6, aspect=1.3)
    plt.savefig(os.path.join(save_dir, "violin_auc_epochs.png"), bbox_inches="tight")

    sns.catplot(x="Distribution", y="AUROC", hue="Parameters", data=main_auc_df, kind="box", height=6, aspect=1.3)
    plt.savefig(os.path.join(save_dir, "box_auc_epochs.png"), bbox_inches="tight")

    sns.catplot(x="Distribution", y="AP", hue="Parameters", data=main_ap_df, kind="violin", height=6, aspect=1.3)
    plt.savefig(os.path.join(save_dir, "violin_ap_epochs.png"), bbox_inches="tight")

    sns.catplot(x="Distribution", y="AP", hue="Parameters", data=main_ap_df, kind="box", height=6, aspect=1.3)
    plt.savefig(os.path.join(save_dir, "box_ap_epochs.png"), bbox_inches="tight")

    print("✅ Training + full metrics + plots completed successfully")
