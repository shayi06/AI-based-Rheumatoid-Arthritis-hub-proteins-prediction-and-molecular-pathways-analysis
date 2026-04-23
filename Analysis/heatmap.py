"""
Heatmap Construction — Novel Predicted & Known RA Proteins
FIXES APPLIED:
  1. Genes sorted by mean log2FC (most upregulated top, most downregulated bottom)
  2. log2FC values annotated inside each cell
  3. X-axis labels include tissue type: e.g. "GSE55235 (Synovial)"
  4. Figure dimensions widened per panel so cells are readable
  5. Y-axis label corrected to "Gene Symbol"
  6. Colorbar cleanly positioned to the right of Panel B only
"""

import os
import warnings
import GEOparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ==============================
# Paths & Settings
# ==============================
BASE_DIR   = "D:/abi/abi/pythonProject/Abi 2025/Research/GNN_RA/DEG analysis/"
OUTPUT_DIR = os.path.join(BASE_DIR, "final_results/")
ML_FILE    = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\prediction\predicted_ra_proteins_rf_novel_only_with_genes.xlsx"
KNOWN_FILE = r"D:\abi\abi\pythonProject\Abi 2025\Research\network\seed_retrieving\ra_133_proteins.xlsx"

ADJ_P_THRESH = 0.05
FC_THRESH    = 0.5
DPI          = 600

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Dataset configs
# FIX 3: tissue type added to display label
# ==============================
DATASETS = {
    "GSE55235": {
        "ra_kw":      "rheumatoid arthritis",
        "ctrl_kw":    "healthy control",
        "char_field": "disease state",
        "label":      "GSE55235\n(Synovial)",   # <-- display label on x-axis
    },
    "GSE55457": {
        "ra_kw":      "rheumatoid arthritis",
        "ctrl_kw":    "normal control",
        "char_field": "disease state",
        "label":      "GSE55457\n(Synovial)",
    },
    "GSE12021": {
        "ra_kw":      "rheumatoid arthritis",
        "ctrl_kw":    "normal",
        "char_field": "disease state",
        "label":      "GSE12021\n(PBMC)",
    },
    "GSE77298": {
        "ra_kw":      "rheumatoid arthritis",
        "ctrl_kw":    "healthy",
        "char_field": "disease state",
        "label":      "GSE77298\n(Synovial)",
    },
}

# ==============================
# Load ML predicted genes (novel)
# ==============================
ml_df    = pd.read_excel(ML_FILE)
ml_genes = set()
for val in ml_df["Gene_Symbol"].dropna():
    for g in str(val).split("///"):
        g = g.strip().upper()
        if g:
            ml_genes.add(g)
print(f"ML predicted genes loaded: {len(ml_genes)}")

# ==============================
# Load known seed RA genes
# ==============================
known_df = pd.read_excel(KNOWN_FILE)
known_gene_col = None
for col in ["Gene_Symbol", "gene_symbol", "Gene", "gene", "Symbol", "symbol"]:
    if col in known_df.columns:
        known_gene_col = col
        break
if known_gene_col is None:
    raise ValueError(f"Cannot find gene column in known file. Columns: {known_df.columns.tolist()}")

known_genes = set()
for val in known_df[known_gene_col].dropna():
    for g in str(val).split("///"):
        g = g.strip().upper()
        if g:
            known_genes.add(g)
print(f"Known seed genes loaded: {len(known_genes)}")


# ==============================
# Helper — get sample groups
# ==============================
def get_sample_groups(gse, ra_kw, ctrl_kw, char_field):
    ra_samples, ctrl_samples = [], []
    for gsm_name, gsm in gse.gsms.items():
        raw_chars  = gsm.metadata.get("characteristics_ch1", [])
        chars_flat = []
        for c in raw_chars:
            if isinstance(c, list):
                chars_flat.extend([str(x) for x in c])
            else:
                chars_flat.append(str(c))

        status = None
        for c in chars_flat:
            if char_field.lower() in c.lower():
                status = c.split(":")[-1].strip().lower()
                break

        if status is None:
            fallback = " ".join(
                [str(x) for x in gsm.metadata.get("title", [])] +
                [str(x) for x in gsm.metadata.get("source_name_ch1", [])]
            ).lower()
            if ra_kw.lower() in fallback:
                status = ra_kw.lower()
            elif ctrl_kw.lower() in fallback:
                status = ctrl_kw.lower()

        if status is not None:
            if ra_kw.lower() in status:
                ra_samples.append(gsm_name)
            elif ctrl_kw.lower() in status:
                ctrl_samples.append(gsm_name)

    return ra_samples, ctrl_samples


# ==============================
# Helper — build expression matrix
# ==============================
def build_expr_matrix(gse, gpl_name, all_samples):
    gpl      = gse.gpls[gpl_name]
    gene_col = None
    for col in ["Gene Symbol", "GENE_SYMBOL", "gene_symbol", "Symbol",
                "SYMBOL", "ILMN_Gene", "gene_name", "Gene_Symbol", "EntrezGeneSymbol"]:
        if col in gpl.table.columns:
            gene_col = col
            break
    if gene_col is None:
        raise ValueError(f"Cannot find gene symbol column in {gpl_name}. "
                         f"Available: {gpl.table.columns.tolist()}")
    print(f"  Using gene column: '{gene_col}'")

    annot = (
        gpl.table[["ID", gene_col]]
        .rename(columns={"ID": "probe_id", gene_col: "gene_symbol"})
        .dropna(subset=["gene_symbol"])
    )
    annot = annot[annot["gene_symbol"].str.strip() != ""]

    expr_data = {}
    for gsm_name in all_samples:
        gsm  = gse.gsms[gsm_name]
        data = gsm.table[["ID_REF", "VALUE"]].rename(
            columns={"ID_REF": "probe_id", "VALUE": gsm_name}
        )
        data[gsm_name] = pd.to_numeric(data[gsm_name], errors="coerce")
        expr_data[gsm_name] = data.set_index("probe_id")[gsm_name]

    expr_df = (
        pd.DataFrame(expr_data)
        .reset_index()
        .rename(columns={"index": "probe_id"})
        .merge(annot, on="probe_id", how="inner")
        .drop(columns=["probe_id"])
    )
    expr_df["gene_symbol"] = (
        expr_df["gene_symbol"].str.split("///").str[0].str.strip().str.upper()
    )
    expr_gene = expr_df.groupby("gene_symbol").mean()

    vals       = expr_gene.values
    vals_clean = vals[~np.isnan(vals)]
    if len(vals_clean) > 0 and vals_clean.max() > 100:
        expr_gene = np.log2(expr_gene + 1)

    return expr_gene


# ==============================
# Helper — DEG analysis
# ==============================
def run_deg(expr_gene, ra_samples, ctrl_samples):
    ra_cols   = [s for s in ra_samples   if s in expr_gene.columns]
    ctrl_cols = [s for s in ctrl_samples if s in expr_gene.columns]

    results = []
    for gene in expr_gene.index:
        ra_vals   = expr_gene.loc[gene, ra_cols].dropna().values
        ctrl_vals = expr_gene.loc[gene, ctrl_cols].dropna().values
        if len(ra_vals) < 2 or len(ctrl_vals) < 2:
            continue
        _, p_val = stats.ttest_ind(ra_vals, ctrl_vals, equal_var=False)
        results.append({
            "Gene":         gene,
            "log2FC":       float(np.mean(ra_vals) - np.mean(ctrl_vals)),
            "p_value":      p_val,
            "mean_RA":      float(np.mean(ra_vals)),
            "mean_Control": float(np.mean(ctrl_vals)),
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    _, adj_pvals, _, _ = multipletests(df["p_value"], method="fdr_bh")
    df["adj_p_value"]  = adj_pvals
    return df.sort_values("adj_p_value").reset_index(drop=True)


# ==============================
# Helper — build log2FC matrix
# FIX 1: genes sorted by mean log2FC across datasets
# FIX 3: columns renamed to tissue-type labels
# ==============================
def build_log2fc_matrix(gene_set, all_deg_results, dataset_names, dataset_labels):
    validated_genes = set()
    for ds, deg_df in all_deg_results.items():
        if deg_df.empty:
            continue
        sig     = deg_df[
            (deg_df["adj_p_value"] < ADJ_P_THRESH) &
            (deg_df["log2FC"].abs() > FC_THRESH)
        ]
        overlap = set(sig["Gene"].str.upper()).intersection(gene_set)
        validated_genes.update(overlap)

    if not validated_genes:
        return None

    # Build matrix using GSE IDs as columns first
    matrix = pd.DataFrame(index=sorted(validated_genes), columns=dataset_names, dtype=float)

    for ds, deg_df in all_deg_results.items():
        if deg_df.empty:
            continue
        deg_indexed = deg_df.set_index("Gene")
        for gene in validated_genes:
            if gene in deg_indexed.index:
                row = deg_indexed.loc[gene]
                if isinstance(row, pd.DataFrame):
                    row = row.sort_values("adj_p_value").iloc[0]
                is_sig = (
                    row["adj_p_value"] < ADJ_P_THRESH and
                    abs(row["log2FC"]) > FC_THRESH
                )
                matrix.loc[gene, ds] = row["log2FC"] if is_sig else np.nan
            else:
                matrix.loc[gene, ds] = np.nan

    matrix = matrix.dropna(how="all")

    # FIX 1: sort rows by mean log2FC (ignoring NaN), descending
    matrix["_mean_fc"] = matrix.apply(pd.to_numeric, errors="coerce").mean(axis=1)
    matrix = matrix.sort_values("_mean_fc", ascending=False).drop(columns=["_mean_fc"])

    # FIX 3: rename columns to tissue-type labels
    label_map = dict(zip(dataset_names, dataset_labels))
    matrix    = matrix.rename(columns=label_map)

    return matrix


# ==============================
# Helper — plot combined heatmap
# FIX 2: annotate cells with log2FC values
# FIX 4: wider figure, balanced aspect ratio
# FIX 5: y-axis label corrected to "Gene Symbol"
# FIX 6: single colorbar cleanly on the right of Panel B
# ==============================
def plot_combined_heatmap(novel_matrix, known_matrix, output_dir):

    # Shared colour scale
    all_vals = []
    for m in [novel_matrix, known_matrix]:
        if m is not None and not m.empty:
            all_vals.extend(m.values.flatten().tolist())
    all_vals = np.array([v for v in all_vals if not np.isnan(v)])
    vmax = float(np.nanmax(np.abs(all_vals))) if len(all_vals) > 0 else 2.0
    vmin = -vmax

    cmap = sns.diverging_palette(220, 20, as_cmap=True)   # blue → white → red
    cmap.set_bad(color="#D5D8DC")                          # grey = not significant

    # FIX 4: wider figure — each panel gets more horizontal room
    n_rows_novel = len(novel_matrix) if novel_matrix is not None else 0
    n_rows_known = len(known_matrix) if known_matrix is not None else 0
    n_rows       = max(n_rows_novel, n_rows_known)
    fig_height   = max(10, 0.32 * n_rows + 4)
    fig_width    = 22   # was 16 — wider panels, more readable cells

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    def draw_panel(ax, matrix, title, panel_label, show_cbar):
        if matrix is None or matrix.empty:
            ax.text(0.5, 0.5, "No significant DEGs found",
                    ha="center", va="center", fontsize=18, transform=ax.transAxes)
            ax.set_title(title, fontsize=18, fontweight="bold")
            return

        # FIX 2: build annotation array (show value only for non-NaN significant cells)
        annot_data  = matrix.copy().astype(object)
        annot_fmt   = matrix.copy().astype(object)
        for r in matrix.index:
            for c in matrix.columns:
                val = matrix.loc[r, c]
                if pd.isna(val):
                    annot_fmt.loc[r, c] = ""   # grey cell — no annotation
                else:
                    annot_fmt.loc[r, c] = f"{val:.2f}"

        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.3,
            linecolor="#EAECEE",
            annot=annot_fmt,          # FIX 2: annotate with formatted values
            fmt="",                   # values already formatted as strings
            annot_kws={"size": 12, "fontweight": "bold"},
            mask=matrix.isnull(),
            cbar=show_cbar,           # FIX 6: only Panel B gets colorbar
            cbar_kws={
                "label":   "log₂FC (RA vs Control)",
                "shrink":  0.75,
                "aspect":  30,
                "pad":     0.03,
            } if show_cbar else {},
        )

        ax.set_title(title, fontsize=18, fontweight="bold", pad=12)

        # FIX 3: x-axis labels already contain tissue type via column rename
        ax.set_xticklabels(
            ax.get_xticklabels(),
            fontsize=18, rotation=0, ha="center", fontweight="bold"
        )
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
        ax.set_xlabel("Dataset", fontsize=18, fontweight="bold", labelpad=8)
        ax.set_ylabel("Gene Symbol", fontsize=19, fontweight="bold")   # FIX 5

        # Panel label (A / B)
        ax.text(-0.14, 1.04, panel_label,
                transform=ax.transAxes, fontsize=18, fontweight="bold")

    draw_panel(axes[0], novel_matrix, "Predicted RA Proteins",  "A", show_cbar=False)
    draw_panel(axes[1], known_matrix, "Known RA Seed Proteins", "B", show_cbar=True)

    plt.suptitle(
        "Differential Expression Heatmap — RA vs Control\n"
        f"FDR < {ADJ_P_THRESH}  |  |log₂FC| > {FC_THRESH}  |  Grey = not significant",
        fontsize=18, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, "combined_heatmap.png")
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved: {out_path}")


# ==============================
# MAIN — Load GEO, run DEG, plot
# ==============================
all_deg_results = {}

for dataset_name, cfg in DATASETS.items():
    print(f"\n{'='*55}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*55}")

    dest = os.path.join(BASE_DIR, dataset_name + "/")
    os.makedirs(dest, exist_ok=True)

    gse     = GEOparse.get_GEO(geo=dataset_name, destdir=dest, silent=True)
    gpl_key = list(gse.gpls.keys())[0]
    print(f"  GPL: {gpl_key}")

    ra_samples, ctrl_samples = get_sample_groups(
        gse, cfg["ra_kw"], cfg["ctrl_kw"], cfg["char_field"]
    )
    print(f"  RA: {len(ra_samples)}  |  Control: {len(ctrl_samples)}")

    if not ra_samples or not ctrl_samples:
        print(f"  Skipping — sample groups not found.")
        continue

    expr_gene = build_expr_matrix(gse, gpl_key, ra_samples + ctrl_samples)
    print(f"  Genes in matrix: {len(expr_gene)}")

    results_df = run_deg(expr_gene, ra_samples, ctrl_samples)
    if results_df.empty:
        print(f"  No DEG results. Skipping.")
        continue

    all_deg_results[dataset_name] = results_df
    print(f"  DEG analysis complete.")

# ==============================
# Build matrices & plot heatmap
# ==============================
dataset_names  = list(all_deg_results.keys())
dataset_labels = [DATASETS[ds]["label"] for ds in dataset_names]   # FIX 3

print(f"\nDatasets processed successfully: {dataset_names}")

print("Building log2FC matrices...")
novel_matrix = build_log2fc_matrix(ml_genes,    all_deg_results, dataset_names, dataset_labels)
known_matrix = build_log2fc_matrix(known_genes, all_deg_results, dataset_names, dataset_labels)

print(f"  Novel matrix: {novel_matrix.shape if novel_matrix is not None else 'None'}")
print(f"  Known matrix: {known_matrix.shape if known_matrix is not None else 'None'}")

plot_combined_heatmap(novel_matrix, known_matrix, OUTPUT_DIR)
print("\nDONE")