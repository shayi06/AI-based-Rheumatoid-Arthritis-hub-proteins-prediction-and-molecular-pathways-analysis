from transformers import BertTokenizer, BertModel
import torch
import os
import numpy as np
from torch_geometric.data import Data

dist_map_dir = r'D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\Data_preprocess\alpha_cmaps_from_cif'
out_dir = 'alpha_cmap_graph_datas'
os.makedirs(out_dir, exist_ok=True)
print("Saving to:", out_dir)

distance_threshold = 10.0

tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
model = BertModel.from_pretrained('Rostlab/prot_bert_bfd')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()


def clean_sequence(seq):
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

    removed = [aa for aa in seq if aa not in valid_aa]
    cleaned = ''.join([aa for aa in seq if aa in valid_aa])

    print("Removed characters:", set(removed))
    print("Removed count:", len(removed))

    return cleaned

def seq2protbert(seq):
    seq = clean_sequence(seq)
    print("Cleaned seq length:", len(seq))

    if len(seq) < 3:
        print("❌ Too short for ProtBERT")
        return None

    seq = ' '.join(seq)
    inputs = tokenizer(
        seq,
        return_tensors='pt',
        add_special_tokens=True,
        padding=False,
        truncation=True
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state

    seq_len = (attention_mask[0] == 1).sum().item()


    if seq_len <= 2:
        return None

    # remove [CLS] and [SEP]
    features = embeddings[0][1:seq_len - 1].cpu()
    return features

def contact_map_to_edge_index(contact_map):
    row, col = torch.nonzero(contact_map, as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index

def extract_ContactMap_SeqEmbedds(npz_file_name):
    chain_id = os.path.basename(npz_file_name).split('.')[0]


    # Try to load the .npz file
    try:
        npz = np.load(npz_file_name, allow_pickle=True)
    except Exception as e:
        print(f"Skipping {npz_file_name}: cannot load file ({e})")
        return

    # Try accessing required keys safely
    try:
        dist_map = npz['C_alpha']
        seq = npz['seqres']
    except Exception as e:
        print(f"Skipping {npz_file_name}: missing or unreadable required arrays ({e})")
        return

    # Debug seqres type safely
    if np.size(seq) > 0:
        print(f"\n{chain_id}")
        print("seqres type:", type(seq), "element type:", type(seq[0]) if np.ndim(seq) > 0 else "scalar")
    else:
        print(f"\n{chain_id}")
        print("seqres is empty or invalid:", seq)
        print(f"❌ Skipping {chain_id}: seqres empty or invalid")
        return

    try:
        seq = ''.join(seq.tolist())
    except Exception as e:
        print(f"Skipping {npz_file_name}: error processing sequence ({e})")
        return

    node_features = seq2protbert(seq)
    if node_features is None:
        print(f"❌ Skipping {chain_id}: ProtBERT returned None")
        return

    contact_map = (dist_map <= distance_threshold).astype(int)
    contact_map = torch.from_numpy(contact_map)

    print(f"{chain_id}: contact_map={contact_map.shape[0]}, seq_len={len(seq)}")

    if contact_map.shape[0] != len(seq):
        print(f"❌ Skipping {chain_id} due to shape mismatch")
        return

    edge_index = contact_map_to_edge_index(contact_map)
    if edge_index.numel() == 0:
        print(f"❌ Skipping {chain_id}: no edges in graph")
        return
    data = Data(x=node_features, edge_index=edge_index)

    out_path = os.path.join(out_dir, f'{chain_id}.pt')
    torch.save(data, out_path)
    print(f"✅ Saved: {out_path}")


#############################################################################################################
npz_file_list = [f for f in os.listdir(dist_map_dir) if os.path.isfile(os.path.join(dist_map_dir, f))]

for file in npz_file_list:
    file_path = os.path.join(dist_map_dir, file)
    extract_ContactMap_SeqEmbedds(npz_file_name=file_path)
    print(file, ' done ************************************\n')
