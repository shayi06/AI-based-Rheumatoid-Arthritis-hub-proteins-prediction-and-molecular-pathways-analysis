#!/usr/bin/env python
"""
Contact Map Preprocessing Pipeline
Adapted from DeepFRI's proc_pdb_files.py reference implementation.
Replicates biotoolbox.DistanceMapBuilder behaviour using BioPython only.
"""

import os
import numpy as np
from Bio.PDB import MMCIFParser, is_aa, PPBuilder
import multiprocessing
from functools import partial

# -------------------- Paths --------------------
seqres_file = r'D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\structure-n-seq\pdb_seqres.txt'
cif_dir     = r'D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\structure-n-seq\cif_files'
out_dir     = r'D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\structure-n-seq\cmap'
n_threads   = 50
os.makedirs(out_dir, exist_ok=True)


# -------------------- Load SEQRES --------------------
def read_fasta(filename):
    prot2seq = {}
    with open(filename) as f:
        pid, seq = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if pid:
                    prot2seq[pid] = ''.join(seq)
                pid = line[1:].split()[0].replace('_', '-').upper()
                seq = []
            else:
                seq.append(line)
        if pid:
            prot2seq[pid] = ''.join(seq)
    return prot2seq


# -------------------- Replicate retrieve_pdb / DistanceMapBuilder --------------------
def build_maps(structure, seqres, chain_id_target):
    """
    Replicates DeepFRI reference:
        structure_container = build_structure_container_for_pdb(...).with_seqres(sequence)
        mapper = DistanceMapBuilder(atom="CA", glycine_hack=-1)
        ca = mapper.generate_map_for_pdb(structure_container)
        cb = mapper.set_atom("CB").generate_map_for_pdb(structure_container)

    Key behaviours matched:
      - SEQRES drives the output length (seq_len x seq_len)
      - ATOM records aligned to SEQRES by offset from first observed residue
      - Missing residues → nan coords (produce nan distances, not wrong distances)
      - Glycine Cb fallback → Ca coords (glycine_hack=-1 means use Ca position)
    """
    model = structure[0]
    seq_len = len(seqres)

    for chain in model:
        if chain.id != chain_id_target:
            continue

        # Build resseq -> residue lookup (ATOM records only)
        resseq_dict = {}
        for r in chain:
            if is_aa(r, standard=True):
                resseq_dict[r.get_id()[1]] = r

        if not resseq_dict:
            return None

        # Offset: first observed ATOM resseq maps to SEQRES position 0
        # This matches .with_seqres() behaviour in biotoolbox
        min_resseq = min(resseq_dict.keys())

        ca_coords = []
        cb_coords = []

        for i in range(seq_len):
            target_resseq = min_resseq + i
            r = resseq_dict.get(target_resseq)  # None if residue missing from ATOM

            if r is not None:
                # CA
                ca = r['CA'].get_vector().get_array() if 'CA' in r else np.full(3, np.nan, dtype=np.float32)

                # CB — glycine_hack=-1: use CA coords for Gly (same as reference)
                if 'CB' in r:
                    cb = r['CB'].get_vector().get_array()
                else:
                    cb = ca.copy()   # Gly fallback = Ca (not nan)
            else:
                # Residue missing from structure entirely
                ca = np.full(3, np.nan, dtype=np.float32)
                cb = np.full(3, np.nan, dtype=np.float32)

            ca_coords.append(ca)
            cb_coords.append(cb)

        ca_coords = np.array(ca_coords, dtype=np.float32)
        cb_coords = np.array(cb_coords, dtype=np.float32)

        # Euclidean distance matrices — matches DistanceMapBuilder output
        ca_map = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)
        cb_map = np.linalg.norm(cb_coords[:, None, :] - cb_coords[None, :, :], axis=-1)

        return {'C_alpha': ca_map, 'C_beta': cb_map}

    return None


# -------------------- Replicate write_annot_npz --------------------
def process_chain(cif_file, prot2seq, out_dir):
    """
    Replicates DeepFRI reference write_annot_npz + retrieve_pdb.
    Returns (saved, skipped_parse, skipped_no_seqres, skipped_map_fail).
    """
    pdb_id = os.path.basename(cif_file).replace('.cif', '').upper()

    saved             = 0
    skipped_parse     = 0
    skipped_no_seqres = 0
    skipped_map_fail  = 0

    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('X', cif_file)
        model = structure[0]
    except Exception as e:
        print(f"❌ Failed to parse {pdb_id}: {e}")
        skipped_parse += 1
        return (saved, skipped_parse, skipped_no_seqres, skipped_map_fail)

    for chain in model:
        chain_id = chain.id
        prot     = f"{pdb_id}-{chain_id}"

        # Matches reference: prot2seq[prot] lookup
        seqres = prot2seq.get(prot)
        if seqres is None:
            print(f"⚠️  No seqres for {prot} — skipping")
            skipped_no_seqres += 1
            continue

        out_path = os.path.join(out_dir, f"{prot}.npz")
        if os.path.exists(out_path):
            saved += 1
            continue

        try:
            maps = build_maps(structure, seqres, chain_id)
        except Exception as e:
            print(f"❌ map build error {prot}: {e}")
            skipped_map_fail += 1
            continue

        if maps is None:
            print(f"⚠️  No ATOM residues found for {prot}")
            skipped_map_fail += 1
            continue

        expected = len(seqres)
        if maps['C_alpha'].shape != (expected, expected):
            print(f"⚠️  Shape mismatch {prot}: expected ({expected},{expected}), "
                  f"got {maps['C_alpha'].shape}")
            skipped_map_fail += 1
            continue

        # Matches reference np.savez_compressed call exactly:
        #   C_alpha=A_ca, C_beta=A_cb, seqres=prot2seq[prot]
        np.savez_compressed(
            out_path,
            C_alpha=maps['C_alpha'],
            C_beta=maps['C_beta'],
            seqres=seqres            # plain string — matches reference exactly
        )
        saved += 1
        print(f"✅ Saved {prot}.npz")

    return (saved, skipped_parse, skipped_no_seqres, skipped_map_fail)


# -------------------- Main --------------------
if __name__ == "__main__":
    prot2seq  = read_fasta(seqres_file)
    cif_files = [os.path.join(cif_dir, f)
                 for f in os.listdir(cif_dir)
                 if f.endswith('.cif')]

    print(f"Total CIF files found:  {len(cif_files)}")
    print(f"Total SEQRES entries:   {len(prot2seq)}")

    nprocs = min(n_threads, multiprocessing.cpu_count())
    print(f"Using {nprocs} parallel processes")

    if nprocs > 1:
        pool    = multiprocessing.Pool(nprocs)
        results = pool.map(
            partial(process_chain, prot2seq=prot2seq, out_dir=out_dir),
            cif_files
        )
        pool.close()
        pool.join()
    else:
        results = [process_chain(f, prot2seq, out_dir) for f in cif_files]

    results = [r for r in results if r is not None]

    print(f"\n========= FINAL SUMMARY =========")
    print(f"CIF files submitted:        {len(cif_files)}")
    print(f".npz files saved:           {sum(r[0] for r in results)}")
    print(f"Skipped — parse failure:    {sum(r[1] for r in results)}")
    print(f"Skipped — no SEQRES match:  {sum(r[2] for r in results)}")
    print(f"Skipped — map build fail:   {sum(r[3] for r in results)}")
    print(f"=================================")
    print("✅ CMAP preprocessing complete")