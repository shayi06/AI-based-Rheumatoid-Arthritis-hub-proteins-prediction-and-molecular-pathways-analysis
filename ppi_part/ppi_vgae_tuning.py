import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, VGAE
from torch_geometric.utils import negative_sampling
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# ----------------------------
# Load TSV and split rows
# ----------------------------
file_path = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN\9606.protein.links.v12.0.txt"
df = pd.read_csv(file_path, sep=' ')
print(f"Total edges: {len(df)}")

# Split 80% train, 10% val, 10% test
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}, Test rows: {len(test_df)}")

# Save TSV splits
train_file = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN\ppi part\train_tune\train_split.tsv"
val_file   = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN\ppi part\train_tune\val_split.tsv"
test_file  = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN\ppi part\train_tune\test_split.tsv"

train_df.to_csv(train_file, sep='\t', index=False)
val_df.to_csv(val_file, sep='\t', index=False)
test_df.to_csv(test_file, sep='\t', index=False)

# Wrap in lists for training loop
train_files = [train_file]
val_files = [val_file]
test_files = [test_file]

# ----------------------------
# Paths and device
# ----------------------------
directory_path = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN\ppi part\train_tune"
node2vec_file = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN\ppi part\NOde2vec\node2vec_ppi_embedds_dict.pkl"
model_save_path = os.path.join(directory_path, "ppi_vgae.pt")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ----------------------------
# Load Node2Vec embeddings
# ----------------------------
with open(node2vec_file, 'rb') as f:
    node2vec_embeddings = pickle.load(f)

def genDataObjFromTSV(tsv, node2vec_dict):
    df = pd.read_csv(tsv, sep='\t')

    col1 = df['protein1'].to_list()
    col2 = df['protein2'].to_list()
    full_list = col1 + col2
    unique_list = list(dict.fromkeys(full_list))

    # Only keep proteins that have embeddings
    unique_list = [prot for prot in unique_list if prot in node2vec_dict]

    ind2node = {index: item for index, item in enumerate(unique_list)}
    node2ind = {v: k for k, v in ind2node.items()}

    df = df[df['protein1'].isin(node2ind) & df['protein2'].isin(node2ind)]
    df['protein1'] = df['protein1'].map(node2ind)
    df['protein2'] = df['protein2'].map(node2ind)

    edge_index = np.array([df['protein1'].to_list(), df['protein2'].to_list()])
    edge_index = torch.from_numpy(edge_index).long()
    node_features = np.array([node2vec_dict[prot] for prot in unique_list])
    node_features = torch.tensor(node_features, dtype=torch.float)
    data = Data(x=node_features, edge_index=edge_index)
    return data


#####################################################################
class PPIEncoder1(torch.nn.Module):
    def __init__(self, in_size, mid_size, out_size):
        super(PPIEncoder1, self).__init__()
        self.conv_mu = SAGEConv(in_size, out_size)
        self.conv_logstd = SAGEConv(in_size, out_size)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class PPIEncoder2(torch.nn.Module):
    def __init__(self, in_size, mid_size, out_size):
        super(PPIEncoder2, self).__init__()
        self.conv1 = SAGEConv(in_size, mid_size)
        self.conv_mu = SAGEConv(mid_size, out_size)
        self.conv_logstd = SAGEConv(mid_size, out_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class PPIEncoder3(torch.nn.Module):
    def __init__(self, in_size, mid_size, out_size):
        super(PPIEncoder3, self).__init__()
        self.conv1 = SAGEConv(in_size, mid_size)
        self.conv2 = SAGEConv(mid_size, mid_size)
        self.conv_mu = SAGEConv(mid_size, out_size)
        self.conv_logstd = SAGEConv(mid_size, out_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# parameters
out_channels = 200
num_features = 256
mid_channels = 220
epochs = 50

# Model summary
def print_model_summary(model):
    print("Model Summary:")
    print(model)

##########################################
def train(train_data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x.to(device), train_data.edge_index.to(device))
    loss = model.recon_loss(z, train_data.edge_index.to(device))
    loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

def test(test_val_data):
    model.eval()
    neg_edge_index = negative_sampling(test_val_data.edge_index.to(device))
    with torch.no_grad():
        z = model.encode(test_val_data.x.to(device), test_val_data.edge_index.to(device))
    loss = model.recon_loss(z, test_val_data.edge_index.to(device))
    loss = loss + (1 / test_val_data.num_nodes) * model.kl_loss()
    auc, ap = model.test(z, test_val_data.edge_index.to(device), neg_edge_index)
    return auc, ap, float(loss)


#############################################################
# ----------------------------
# Keep everything above same
# ----------------------------

main_auc_tuning_df = pd.DataFrame()
main_ap_tuning_df = pd.DataFrame()

for lr in [0.1, 0.01, 0.001]:
    for i, encoder in enumerate([PPIEncoder1, PPIEncoder2, PPIEncoder3]):
        model = VGAE(encoder(num_features, mid_channels, out_channels))
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        print_model_summary(model)
        print(f"Learning rate: {lr}, Encoder: {i+1}")

        # Lists to store epoch-wise metrics
        epoch_test_aucs, epoch_test_aps = [], []
        epoch_val_aucs, epoch_val_aps = [], []

        for epoch in range(epochs):
            train_data = genDataObjFromTSV(train_files[0], node2vec_embeddings)
            trloss = train(train_data.to(device))

            # Testing
            test_aucs, test_aps = [], []
            for test_data_file in test_files:
                test_data = genDataObjFromTSV(test_data_file, node2vec_embeddings)
                test_data = test_data.to(device)
                test_auc, test_ap, _ = test(test_data)
                test_aucs.append(test_auc)
                test_aps.append(test_ap)
            mean_test_auc = np.mean(test_aucs)
            mean_test_ap = np.mean(test_aps)

            # Validation
            val_aucs, val_aps = [], []
            for val_data_file in val_files:
                val_data = genDataObjFromTSV(val_data_file, node2vec_embeddings)
                val_data = val_data.to(device)
                val_auc, val_ap, _ = test(val_data)
                val_aucs.append(val_auc)
                val_aps.append(val_ap)
            mean_val_auc = np.mean(val_aucs)
            mean_val_ap = np.mean(val_aps)

            print(f"Epoch {epoch+1}/{epochs} | Train loss: {trloss:.4f} | "
                  f"Test AUROC: {mean_test_auc:.4f} | Val AUROC: {mean_val_auc:.4f} | "
                  f"Test AP: {mean_test_ap:.4f} | Val AP: {mean_val_ap:.4f}")

            # Append epoch-wise metrics
            epoch_test_aucs.append(mean_test_auc)
            epoch_test_aps.append(mean_test_ap)
            epoch_val_aucs.append(mean_val_auc)
            epoch_val_aps.append(mean_val_ap)

        # After all epochs, create DataFrames for plotting
        auc_df = pd.DataFrame({
            "AUROC": epoch_test_aucs + epoch_val_aucs,
            "Distribution": ["Test"]*len(epoch_test_aucs) + ["Validation"]*len(epoch_val_aucs),
            "Epoch": list(range(1, epochs+1))*2,
            "Parameters": [f'GSAGE layers-{i+1}, lr-{lr}']*(2*epochs)
        })
        main_auc_tuning_df = pd.concat([main_auc_tuning_df, auc_df], ignore_index=True)

        ap_df = pd.DataFrame({
            "AP": epoch_test_aps + epoch_val_aps,
            "Distribution": ["Test"]*len(epoch_test_aps) + ["Validation"]*len(epoch_val_aps),
            "Epoch": list(range(1, epochs+1))*2,
            "Parameters": [f'GSAGE layers-{i+1}, lr-{lr}']*(2*epochs)
        })
        main_ap_tuning_df = pd.concat([main_ap_tuning_df, ap_df], ignore_index=True)

# ----------------------------
# Plotting
# ----------------------------
sns.set(style="whitegrid")

sns.catplot(x="Distribution", y="AUROC", data=main_auc_tuning_df,
            kind='violin', hue='Parameters', height=6, aspect=1.2)
plt.title('ROC AUC Distributions for Testing And Validation Datasets')
plt.savefig('violinplot_roc.png', bbox_inches='tight')

sns.catplot(x="Distribution", y="AUROC", data=main_auc_tuning_df,
            kind='box', hue='Parameters', height=6, aspect=1.2)
plt.title('ROC AUC Distributions for Testing And Validation Datasets')
plt.savefig('boxplot_roc.png', bbox_inches='tight')

sns.catplot(x="Distribution", y="AP", data=main_ap_tuning_df,
            kind='violin', hue='Parameters', height=6, aspect=1.2)
plt.title('AP Distributions for Testing And Validation Datasets')
plt.savefig('violinplot_ap.png', bbox_inches='tight')

sns.catplot(x="Distribution", y="AP", data=main_ap_tuning_df,
            kind='box', hue='Parameters', height=6, aspect=1.2)
plt.title('AP Distributions for Testing And Validation Datasets')
plt.savefig('boxplot_ap.png', bbox_inches='tight')
