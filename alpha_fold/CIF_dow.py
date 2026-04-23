import csv
import os
import requests

ensp_uniprot_file = r'D:\abi\abi\pythonProject\Abi 2025\Research\GNN\ppi part\NOde2vec\Ensp_uniID.tsv'
pdb_uniprot_file = r'D:\abi\abi\pythonProject\Abi 2025\Research\GNN\data_preprocessing\pdb_chain_uniprot.csv'
seq_file = r'D:\abi\abi\pythonProject\Abi 2025\Research\GNN\data_preprocessing\pdb_seqres.txt'

# ----------------- 1️⃣ Read reviewed UniProt IDs -----------------
reviewed_uniprot_ids = set()
with open(ensp_uniprot_file, 'r', newline='') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        uniprot = row['Entry'].strip().upper()
        is_reviewed = row['Reviewed'].strip().lower()
        if uniprot and is_reviewed == 'reviewed':
            reviewed_uniprot_ids.add(uniprot)

print(f"Total reviewed UniProt IDs: {len(reviewed_uniprot_ids)}")

# ----------------- 2️⃣ Check which reviewed UniProt IDs are in pdb_chain_uniprot.csv -----------------
pdb_ids = set()
with open(pdb_uniprot_file, 'r', newline='') as f:
    lines = (line for line in f if not line.startswith('#'))
    reader = csv.DictReader(lines)
    for row in reader:
        uniprot = row['SP_PRIMARY'].strip().upper()
        pdb = row['PDB'].strip().upper()
        if uniprot in reviewed_uniprot_ids:
            pdb_ids.add(pdb)

print(f"Total PDB IDs corresponding to reviewed UniProt IDs: {len(pdb_ids)}")

# ----------------- 3️⃣ Check which PDB IDs are in seq.txt -----------------
seq_pdb_ids = set()
with open(seq_file, 'r') as f:
    for line in f:
        if line.startswith('>'):
            header = line[1:].strip()
            pdb_id = header.split('_')[0].upper()
            seq_pdb_ids.add(pdb_id)

existing_pdb_ids = pdb_ids.intersection(seq_pdb_ids)
print(f"Total PDB IDs from reviewed UniProt present in seq.txt: {len(existing_pdb_ids)}")

# ----------------- 4️⃣ Download CIF files -----------------
cif_dir = r'D:\abi\abi\pythonProject\Abi 2025\Research\GNN\data_preprocessing\cif_files'
os.makedirs(cif_dir, exist_ok=True)

for pdb_id in existing_pdb_ids:
    pdb_lower = pdb_id.lower()
    url = f'https://files.rcsb.org/download/{pdb_lower}.cif'
    save_path = os.path.join(cif_dir, f'{pdb_id}.cif')
    if not os.path.exists(save_path):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(r.content)
                print(f'Downloaded {pdb_id}.cif')
            else:
                print(f'Failed to download {pdb_id}: HTTP {r.status_code}')
        except Exception as e:
            print(f'Error downloading {pdb_id}: {e}')

print('Completed downloading CIF files.')
