from transformers import BertTokenizer, BertModel
import torch
import os
import numpy as np
from torch_geometric.data import Data

# ======================== Paths ========================
dist_map_dir = '/home/hpc_users/.../cmaps/'
out_dir      = '/home/hpc_users/.../cmap_graph_datas/'
distance_threshold = 10.0

os.makedirs(out_dir, exist_ok=True)

# ======================== Model ========================
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
bert_model = BertModel.from_pretrained('Rostlab/prot_bert_bfd')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
bert_model.to(device).eval()

# ======================== Helpers ========================

def load_seqres(npz):
    seq = npz['seqres']
    if isinstance(seq, np.ndarray):
        return ''.join(seq.tolist())
    return seq


def clean_sequence(seq):
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    return ''.join([aa for aa in seq if aa in valid])


# 🔥 FIXED ProtBERT (NO truncation, single sequence)
def seq2protbert(seq):
    seq_spaced = ' '.join(seq)

    inputs = tokenizer(
        seq_spaced,
        return_tensors='pt',
        add_special_tokens=True,
        padding=False,
        truncation=False   # 🚨 IMPORTANT
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        embeddings = bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

    seq_len = (attention_mask[0] == 1).sum().item()

    if seq_len <= 2:
        return None

    # remove CLS + SEP
    return embeddings[0][1:seq_len-1].cpu()


def contact_map_to_edge_index(contact_map):
    row, col = torch.nonzero(contact_map, as_tuple=True)
    return torch.stack([row, col], dim=0)


# ======================== Main ========================

def extract_graph(npz_file_path):
    chain_id = os.path.basename(npz_file_path).split('.')[0]
    out_path = os.path.join(out_dir, f'{chain_id}.pt')

    if os.path.exists(out_path):
        return

    # Load npz
    try:
        npz = np.load(npz_file_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ Cannot load {chain_id}: {e}")
        return

    # Extract data
    try:
        dist_map = npz['C_alpha']
        seq = load_seqres(npz)
    except Exception as e:
        print(f"❌ Missing keys in {chain_id}: {e}")
        return

    if not seq:
        print(f"❌ Skipping {chain_id}: empty sequence")
        return

    # 🔥 SHAPE CHECK FIRST
    if dist_map.shape[0] != len(seq):
        print(f"❌ Skipping {chain_id}: dist_map {dist_map.shape[0]} != seq {len(seq)}")
        return

    # 🔥 CLEAN AA CHECK
    seq_clean = clean_sequence(seq)
    if len(seq_clean) != len(seq):
        print(f"❌ Skipping {chain_id}: non-standard AAs found")
        return

    if len(seq_clean) < 3:
        print(f"❌ Skipping {chain_id}: too short")
        return

    # 🔥 ProtBERT embeddings
    node_features = seq2protbert(seq_clean)

    if node_features is None:
        print(f"❌ Skipping {chain_id}: embedding failed")
        return

    # 🔥 Contact map (DeepFRI style)
    contact_map = torch.from_numpy(
        (dist_map <= distance_threshold).astype(np.uint8)
    )

    # 🔥 FINAL ALIGNMENT CHECK
    if contact_map.shape[0] != node_features.shape[0]:
        print(f"❌ Skipping {chain_id}: cmap != embedding size")
        return

    edge_index = contact_map_to_edge_index(contact_map)

    if edge_index.numel() == 0:
        print(f"❌ Skipping {chain_id}: no edges")
        return

    data = Data(x=node_features, edge_index=edge_index)
    torch.save(data, out_path)

    print(f"✅ {chain_id}.pt  [nodes={node_features.shape[0]}, edges={edge_index.shape[1]}]")


# ======================== Run ========================

npz_files = [
    os.path.join(dist_map_dir, f)
    for f in os.listdir(dist_map_dir)
    if f.endswith('.npz')
]

print(f"Total .npz files: {len(npz_files)}")

for npz_path in npz_files:
    extract_graph(npz_path)

print("\n✅ Graph building complete")