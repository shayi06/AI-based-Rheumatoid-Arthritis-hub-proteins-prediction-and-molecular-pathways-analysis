import tarfile
import os
import shutil
import gzip

# ---------------------- Paths ----------------------
tar_file = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\Data_preprocess\UP000005640_9606_HUMAN_v6.tar"
cif_gz_dir = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\Data_preprocess\alpha_cif_files"
os.makedirs(cif_gz_dir, exist_ok=True)

# ---------------------- Extract only .cif.gz ----------------------
print("Extracting only .cif.gz files from tar...")

with tarfile.open(tar_file, 'r') as tar:
    for member in tar.getmembers():
        if member.name.endswith('.cif.gz'):
            # Extract to temp location
            tar.extract(member, path=cif_gz_dir)
            # Move to top-level folder
            extracted_path = os.path.join(cif_gz_dir, member.name)
            final_path = os.path.join(cif_gz_dir, os.path.basename(member.name))
            shutil.move(extracted_path, final_path)

print(f"✅ Extracted all CIF.GZ files to: {cif_gz_dir}")

# ---------------------- Optional: Decompress CIF.GZ ----------------------
decompress = True  # set True if you want uncompressed .cif
if decompress:
    print("Decompressing CIF.GZ files...")
    for f in os.listdir(cif_gz_dir):
        if f.endswith('.cif.gz'):
            gz_path = os.path.join(cif_gz_dir, f)
            cif_path = os.path.join(cif_gz_dir, f[:-3])
            with gzip.open(gz_path, 'rb') as gz_file:
                with open(cif_path, 'wb') as out_file:
                    shutil.copyfileobj(gz_file, out_file)
            os.remove(gz_path)
    print(f"✅ Decompressed all CIF files in {cif_gz_dir}")
