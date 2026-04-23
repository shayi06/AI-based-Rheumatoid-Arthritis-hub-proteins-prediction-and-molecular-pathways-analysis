#!/usr/bin/env python

import os
import numpy as np
from Bio.PDB import MMCIFParser, is_aa
import multiprocessing
from functools import partial
from pathlib import Path

# -------------------- Paths --------------------
cif_dir   = r'D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\Data_preprocess\alpha_cif_files'
out_dir   = r'D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\Data_preprocess\alpha_cmaps_from_cif'
n_threads = 50
os.makedirs(out_dir, exist_ok=True)

# -------------------- Extract sequence from CIF --------------------
def extract_seq_from_cif(cif_path):
    """
    Extracts sequences for all chains from an AlphaFold CIF.
    Returns: dict {chain_id: sequence}
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("X", cif_path)
    model = structure[0]

    seq_dict = {}
    for chain in model:
        seq = ""
        for res in chain:
            if is_aa(res):
                try:
                    seq += res.get_resname()
                except Exception:
                    seq += "X"
        # Convert 3-letter code to 1-letter
        from Bio.SeqUtils import seq1
        seq_dict[chain.id] = seq1(seq)
    return seq_dict

# -------------------- Build contact maps --------------------
def build_contact_maps(cif_path, seq_dict):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("X", cif_path)
    model = structure[0]

    maps_dict = {}
    for chain in model:
        chain_id = chain.id
        seq = seq_dict.get(chain_id, "")
        if not seq:
            continue

        ca_coords, cb_coords = [], []
        res_list = [r for r in chain if is_aa(r)]

        for r in res_list:
            ca_coords.append(r['CA'].get_vector() if 'CA' in r else np.array([np.nan]*3))
            if 'CB' in r:
                cb_coords.append(r['CB'].get_vector())
            else:
                cb_coords.append(r['CA'].get_vector())

        ca_coords = np.array([v.get_array() if hasattr(v, 'get_array') else v for v in ca_coords])
        cb_coords = np.array([v.get_array() if hasattr(v, 'get_array') else v for v in cb_coords])

        # Distance matrices
        ca_map = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)
        cb_map = np.linalg.norm(cb_coords[:, None, :] - cb_coords[None, :, :], axis=-1)

        maps_dict[chain_id] = {"C_alpha": ca_map, "C_beta": cb_map, "seqres": seq}

    return maps_dict

# -------------------- Process one CIF file --------------------
def process_cif(cif_file, out_dir):
    pdb_name = Path(cif_file).stem
    try:
        seq_dict = extract_seq_from_cif(cif_file)
        maps_dict = build_contact_maps(cif_file, seq_dict)

        for chain_id, mats in maps_dict.items():
            # Extract UniProt ID from filename (AlphaFold CIF naming)
            uniprot_id = os.path.basename(cif_file).split('-')[1]  # e.g., AF-A0A024R1R8-F1-model_v6.cif -> A0A024R1R8
            outfile_name = f"{uniprot_id}-{chain_id}.npz"

            # Save
            np.savez_compressed(
                os.path.join(out_dir, outfile_name),
                C_alpha=mats['C_alpha'],
                C_beta=mats['C_beta'],
                seqres=mats['seqres']
            )

            print(f"✅ Saved {outfile_name}")

    except Exception as e:
        print(f"❌ Failed {pdb_name}: {e}")

# -------------------- Main --------------------
if __name__ == "__main__":
    cif_files = [os.path.join(cif_dir, f)
                 for f in os.listdir(cif_dir)
                 if f.endswith('.cif') and '-F1-' in f]
    nprocs = min(n_threads, multiprocessing.cpu_count())
    if nprocs > 1:
        pool = multiprocessing.Pool(nprocs)
        pool.map(partial(process_cif, out_dir=out_dir), cif_files)
        pool.close()
        pool.join()
    else:
        for f in cif_files:
            process_cif(f, out_dir)

    print("✅ All AlphaFold CIF contact maps saved!")
