"""
DEG Analysis Pipeline — RA Novel Predicted Proteins Validation
Datasets: GSE55235, GSE55457, GSE12021, GSE77298
Fixed version — auto-detects GPL key
Includes: Volcano plots + Venn diagram + Heatmaps (novel & known proteins)
"""

import os
import warnings
import GEOparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib_venn import venn3
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ==============================
# Paths and settings
# ==============================
BASE_DIR     = "D:/abi/abi/pythonProject/Abi 2025/Research/GNN_RA/DEG analysis/"
OUTPUT_DIR   = os.path.join(BASE_DIR, "final_results/")

# Novel predicted RA proteins (217 candidates)
ML_FILE      = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\prediction\predicted_ra_proteins_rf_novel_only_with_genes.xlsx"

# Known seed RA proteins (98 proteins / 133 original seeds)
KNOWN_FILE   = r"D:\abi\abi\pythonProject\Abi 2025\Research\network\seed_retrieving\ra_133_proteins.xlsx"

ADJ_P_THRESH = 0.05
FC_THRESH    = 0.5
DPI          = 600
MAX_GENES    = 60    # Maximum genes to display per heatmap

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Dataset configs
# ==============================
DATASETS = {
    "GSE55235": {
        "ra_kw":      "rheumatoid arthritis",
        "ctrl_kw":    "healthy control",
        "tissue":     "Synovial Tissue",
        "char_field": "disease state",
    },
    "GSE55457": {
        "ra_kw":      "rheumatoid arthritis",
        "ctrl_kw":    "normal control",
        "tissue":     "Synovial Tissue",
        "char_field": "disease state",
    },
    "GSE12021": {
        "ra_kw":      "rheumatoid arthritis",
        "ctrl_kw":    "normal",
        "tissue":     "Synovial Tissue",
        "char_field": "disease state",
    },
    "GSE77298": {
        "ra_kw":      "rheumatoid arthritis",
        "ctrl_kw":    "healthy",
        "tissue":     "PBMC",
        "char_field": "disease state",
    },
}

GEO_DATASET_NAMES = list(DATASETS.keys())

# ==============================
# Load ML predicted proteins (novel)
# ==============================
if not os.path.exists(ML_FILE):
    raise FileNotFoundError(f"ML file not found: {ML_FILE}")

ml_df    = pd.read_excel(ML_FILE)
ml_genes = set()
for val in ml_df["Gene_Symbol"].dropna():
    for g in str(val).split("///"):
        g = g.strip().upper()
        if g:
            ml_genes.add(g)
print(f"ML predicted gene symbols loaded: {len(ml_genes)}")

# ==============================
# Load known seed RA proteins
# ==============================
if not os.path.exists(KNOWN_FILE):
    raise FileNotFoundError(f"Known proteins file not found: {KNOWN_FILE}")

known_df    = pd.read_excel(KNOWN_FILE)

# Auto-detect gene symbol column in known proteins file
known_gene_col = None
for col in ["Gene_Symbol", "gene_symbol", "Gene", "gene", "Symbol", "symbol"]:
    if col in known_df.columns:
        known_gene_col = col
        break

if known_gene_col is None:
    print(f"Available columns in known file: {known_df.columns.tolist()}")
    raise ValueError("Cannot find gene symbol column in known proteins file.")

known_genes = set()
for val in known_df[known_gene_col].dropna():
    for g in str(val).split("///"):
        g = g.strip().upper()
        if g:
            known_genes.add(g)
print(f"Known seed proteins loaded: {len(known_genes)}")


# ==============================
# Helper — inspect sample metadata
# ==============================
def inspect_samples(gse, n=5):
    for i, (gsm_name, gsm) in enumerate(gse.gsms.items()):
        if i >= n:
            break
        chars = gsm.metadata.get("characteristics_ch1", [])
        title = gsm.metadata.get("title", [""])[0]
        src   = gsm.metadata.get("source_name_ch1", [""])[0]
        print(f"  {gsm_name} | title: {title} | src: {src} | chars: {chars}")


# ==============================
# Helper — get RA/control samples
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
# Helper — auto-detect GPL key
# ==============================
def get_gpl_key(gse):
    keys = list(gse.gpls.keys())
    if not keys:
        raise ValueError("No GPL platforms found in this GSE!")
    print(f"  Available GPL keys: {keys}  → using: {keys[0]}")
    return keys[0]


# ==============================
# Helper — build expression matrix
# ==============================
def build_expr_matrix(gse, gpl_name, all_samples):
    gpl      = gse.gpls[gpl_name]
    gene_col = None
    for col in ["Gene Symbol", "GENE_SYMBOL", "gene_symbol",
                "Symbol", "SYMBOL", "ILMN_Gene", "gene_name",
                "Gene_Symbol", "EntrezGeneSymbol"]:
        if col in gpl.table.columns:
            gene_col = col
            break

    if gene_col is None:
        print(f"  Available GPL columns: {gpl.table.columns.tolist()}")
        raise ValueError(f"Cannot find gene symbol column in {gpl_name}")

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
        expr_df["gene_symbol"]
        .str.split("///").str[0]
        .str.strip()
        .str.upper()
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
# Helper — volcano plot
# ==============================
def plot_volcano(results_df, ml_deg_df, dataset_name, tissue,
                 n_ra, n_ctrl, n_degs, output_dir):
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_facecolor("#FAFAFA")

    def point_color(row):
        if row["adj_p_value"] < ADJ_P_THRESH and row["log2FC"] > FC_THRESH:
            return "#E74C3C"
        elif row["adj_p_value"] < ADJ_P_THRESH and row["log2FC"] < -FC_THRESH:
            return "#3498DB"
        return "#BDC3C7"

    colors = results_df.apply(point_color, axis=1)
    ax.scatter(
        results_df["log2FC"],
        -np.log10(results_df["adj_p_value"] + 1e-300),
        c=colors, alpha=0.35, s=8, rasterized=True
    )

    for _, row in ml_deg_df.iterrows():
        col = "#C0392B" if row["log2FC"] > 0 else "#1A5276"
        ax.scatter(
            row["log2FC"],
            -np.log10(row["adj_p_value"] + 1e-300),
            c=col, s=160, marker="D",
            edgecolors="black", linewidths=1.2, zorder=5
        )
        ax.annotate(
            row["Gene"],
            xy=(row["log2FC"], -np.log10(row["adj_p_value"] + 1e-300)),
            xytext=(6, 3), textcoords="offset points",
            fontsize=15, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.75)
        )

    ax.axvline(x= FC_THRESH, color="grey", linestyle="--", alpha=0.5, lw=1)
    ax.axvline(x=-FC_THRESH, color="grey", linestyle="--", alpha=0.5, lw=1)
    ax.axhline(y=-np.log10(ADJ_P_THRESH), color="grey", linestyle="--", alpha=0.5, lw=1)

    up   = ml_deg_df[ml_deg_df["log2FC"] > 0]
    down = ml_deg_df[ml_deg_df["log2FC"] < 0]

    legend_elements = [
        mpatches.Patch(color="#E74C3C", label="Upregulated DEGs"),
        mpatches.Patch(color="#3498DB", label="Downregulated DEGs"),
        mpatches.Patch(color="#BDC3C7", label="Not significant"),
        plt.Line2D([0],[0], marker="D", color="w", markerfacecolor="#C0392B",
                   markersize=9, label=f"ML validated UP ({len(up)})"),
        plt.Line2D([0],[0], marker="D", color="w", markerfacecolor="#1A5276",
                   markersize=9, label=f"ML validated DOWN ({len(down)})"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=12, framealpha=0.8)
    ax.set_xlabel("log2 Fold Change (RA vs Control)", fontsize=18)
    ax.set_ylabel("-log10(adjusted p-value)", fontsize=18)
    ax.set_title(
        f"Volcano Plot — {dataset_name}  |  {tissue}: RA vs Control\n"
        f"adj_p<{ADJ_P_THRESH}, |log2FC|>{FC_THRESH}  |  "
        f"Samples: {n_ra} RA / {n_ctrl} Ctrl  |  "
        f"Total DEGs: {n_degs}  |  ML validated: {len(ml_deg_df)}",
        fontsize=18, fontweight="bold"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_volcano.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Volcano saved: {path}")


# ==============================
# Helper — build log2FC matrix
# ==============================
def build_log2fc_matrix(gene_set, all_deg_results, dataset_names,
                         fdr_threshold, log2fc_threshold):
    """
    Builds a gene x dataset log2FC matrix for genes that:
    - Belong to gene_set (novel predicted OR known proteins)
    - Are significantly differentially expressed in at least one dataset
    Grey (NaN) = gene tested but not significant in that dataset
    """
    # Collect all validated genes across all datasets
    validated_genes = set()
    for ds, deg_df in all_deg_results.items():
        if deg_df.empty:
            continue
        sig = deg_df[
            (deg_df["adj_p_value"] < fdr_threshold) &
            (deg_df["log2FC"].abs() > log2fc_threshold)
        ]
        overlap = set(sig["Gene"].str.upper()).intersection(gene_set)
        validated_genes.update(overlap)

    if not validated_genes:
        return None

    # Build matrix: rows = genes, columns = datasets
    matrix = pd.DataFrame(
        index=sorted(validated_genes),
        columns=dataset_names,
        dtype=float
    )

    for ds, deg_df in all_deg_results.items():
        if deg_df.empty:
            continue
        deg_df_indexed = deg_df.set_index("Gene")
        for gene in validated_genes:
            if gene in deg_df_indexed.index:
                row = deg_df_indexed.loc[gene]
                # Handle duplicate gene entries — take the most significant
                if isinstance(row, pd.DataFrame):
                    row = row.sort_values("adj_p_value").iloc[0]
                is_sig = (
                    row["adj_p_value"] < fdr_threshold and
                    abs(row["log2FC"]) > log2fc_threshold
                )
                # Show log2FC only if significant, NaN otherwise (shown as grey)
                matrix.loc[gene, ds] = row["log2FC"] if is_sig else np.nan
            else:
                matrix.loc[gene, ds] = np.nan

    # Drop genes with no significant values in any dataset
    matrix = matrix.dropna(how="all")
    return matrix


def plot_combined_heatmap(novel_matrix, known_matrix, output_dir):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    # Fill NaN → NO grey / NO white gaps
    novel_matrix = novel_matrix.fillna(0)
    known_matrix = known_matrix.fillna(0)

    # 🔥 Strong color palette (NO white center)
    cmap = sns.color_palette("icefire", as_cmap=True)

    # Shared scale
    all_vals = np.concatenate([
        novel_matrix.values.flatten(),
        known_matrix.values.flatten()
    ])
    vmax = np.max(np.abs(all_vals))
    vmin = -vmax

    # 👉 Smaller cells + bigger fonts
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    def draw(ax, matrix, title, label, show_cbar=False):
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.3,
            linecolor="white",
            annot=False,
            cbar=show_cbar
        )

        ax.set_title(title, fontsize=18, fontweight="bold")

        ax.set_xticklabels(
            ax.get_xticklabels(),
            fontsize=16,
            rotation=40,
            ha="right",
            fontweight="bold"
        )

        ax.set_yticklabels(
            ax.get_yticklabels(),
            fontsize=16
        )

        ax.set_xlabel("Dataset", fontsize=16, fontweight="bold")
        ax.set_ylabel("Genes", fontsize=16, fontweight="bold")

        # Panel label
        ax.text(-0.2, 1.05, label,
                transform=ax.transAxes,
                fontsize=20,
                fontweight="bold")

    draw(axes[0], novel_matrix, "Predicted RA Proteins", "A", False)
    draw(axes[1], known_matrix, "Known RA Proteins", "B", True)

    plt.suptitle(
        "Differential Expression Across Datasets",
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout()

    out_path = os.path.join(output_dir, "combined_heatmap.png")
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {out_path}")

# ==============================
# MAIN LOOP — DEG analysis per dataset
# ==============================
all_deg_sets    = {}
all_ml_deg_dfs  = {}
all_deg_results = {}   # Full DEG results needed for heatmap matrix

for dataset_name, cfg in DATASETS.items():
    print(f"\n{'='*60}")
    print(f"PROCESSING: {dataset_name}")
    print(f"{'='*60}")

    dest = os.path.join(BASE_DIR, dataset_name + "/")
    os.makedirs(dest, exist_ok=True)

    print(f"  Loading {dataset_name}...")
    gse     = GEOparse.get_GEO(geo=dataset_name, destdir=dest, silent=True)
    gpl_key = get_gpl_key(gse)

    ra_samples, ctrl_samples = get_sample_groups(
        gse, cfg["ra_kw"], cfg["ctrl_kw"], cfg["char_field"]
    )
    print(f"  RA samples:      {len(ra_samples)}")
    print(f"  Control samples: {len(ctrl_samples)}")

    if len(ra_samples) == 0 or len(ctrl_samples) == 0:
        print(f"  Skipping {dataset_name} — sample groups not found.")
        continue

    all_samples = ra_samples + ctrl_samples
    print(f"  Building expression matrix...")
    expr_gene = build_expr_matrix(gse, gpl_key, all_samples)
    print(f"  Genes in matrix: {len(expr_gene)}")

    print(f"  Running DEG analysis...")
    results_df = run_deg(expr_gene, ra_samples, ctrl_samples)

    if results_df.empty:
        print(f"  No results. Skipping.")
        continue

    # Store full DEG results for heatmap
    all_deg_results[dataset_name] = results_df

    sig_degs     = results_df[
        (results_df["adj_p_value"] < ADJ_P_THRESH) &
        (results_df["log2FC"].abs() > FC_THRESH)
    ].copy()
    deg_gene_set = set(sig_degs["Gene"].str.upper())

    print(f"  Total DEGs:           {len(sig_degs)}")
    print(f"  Upregulated in RA:    {len(sig_degs[sig_degs.log2FC > 0])}")
    print(f"  Downregulated in RA:  {len(sig_degs[sig_degs.log2FC < 0])}")

    ml_overlap = deg_gene_set.intersection(ml_genes)
    ml_deg_df  = sig_degs[sig_degs["Gene"].str.upper().isin(ml_overlap)].copy()
    ml_deg_df  = ml_deg_df.sort_values("adj_p_value").reset_index(drop=True)
    ml_deg_df["Direction"] = ml_deg_df["log2FC"].apply(
        lambda x: "Upregulated" if x > 0 else "Downregulated"
    )
    ml_deg_df["Dataset"] = dataset_name

    print(f"  Predicted candidates validated: {len(ml_deg_df)}")

    sig_degs.to_csv(
        os.path.join(OUTPUT_DIR, f"{dataset_name}_all_DEGs.csv"), index=False
    )
    ml_deg_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{dataset_name}_ML_validated_DEGs.csv"), index=False
    )

    plot_volcano(
        results_df, ml_deg_df, dataset_name, cfg["tissue"],
        len(ra_samples), len(ctrl_samples), len(sig_degs), OUTPUT_DIR
    )

    all_deg_sets[dataset_name]   = set(ml_deg_df["Gene"].str.upper())
    all_ml_deg_dfs[dataset_name] = ml_deg_df


# ==============================
# CROSS-DATASET OVERLAP
# ==============================
print(f"\n{'='*60}")
print("CROSS-DATASET OVERLAP")
print(f"{'='*60}")

dataset_names = list(all_deg_sets.keys())

if len(dataset_names) < 2:
    print("Less than 2 datasets processed successfully.")
else:
    gene_dataset_map = {}
    for ds, gset in all_deg_sets.items():
        for g in gset:
            gene_dataset_map.setdefault(g, []).append(ds)

    confirmed_1plus = {g for g, ds in gene_dataset_map.items() if len(ds) >= 1}
    confirmed_2plus = {g for g, ds in gene_dataset_map.items() if len(ds) >= 2}
    confirmed_3plus = {g for g, ds in gene_dataset_map.items() if len(ds) >= 3}
    confirmed_all   = {g for g, ds in gene_dataset_map.items() if len(ds) == len(dataset_names)}

    print(f"Predicted candidates validated per dataset:")
    for ds in dataset_names:
        print(f"  {ds}: {len(all_deg_sets[ds])} genes")
    print(f"\nConfirmed in 1+ dataset:  {len(confirmed_1plus)}")
    print(f"Confirmed in 2+ datasets: {len(confirmed_2plus)}")
    print(f"Confirmed in 3+ datasets: {len(confirmed_3plus)}")
    print(f"Confirmed in all {len(dataset_names)}:       {len(confirmed_all)}")

    # Summary CSV
    combined_df  = pd.concat(list(all_ml_deg_dfs.values()), ignore_index=True)
    summary_rows = []
    for gene, datasets in gene_dataset_map.items():
        rows    = combined_df[combined_df["Gene"].str.upper() == gene]
        mean_fc = rows["log2FC"].mean()
        summary_rows.append({
            "Gene":         gene,
            "Datasets":     ", ".join(sorted(datasets)),
            "N_datasets":   len(datasets),
            "Mean_log2FC":  round(mean_fc, 4),
            "Min_adj_pval": round(rows["adj_p_value"].min(), 6),
            "Direction":    "Upregulated" if mean_fc > 0 else "Downregulated",
        })

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["N_datasets", "Min_adj_pval"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "ML_validated_DEGs_summary.csv"), index=False
    )
    print(f"\nTop genes confirmed in 2+ datasets:")
    print(
        summary_df[summary_df["N_datasets"] >= 2][
            ["Gene", "N_datasets", "Mean_log2FC", "Direction"]
        ].head(20).to_string(index=False)
    )

    # Venn diagram (first 3 datasets)
    venn_datasets = dataset_names[:3]
    if len(venn_datasets) == 3:
        s1 = all_deg_sets[venn_datasets[0]]
        s2 = all_deg_sets[venn_datasets[1]]
        s3 = all_deg_sets[venn_datasets[2]]

        fig, ax = plt.subplots(figsize=(9, 7))
        venn3(
            [s1, s2, s3],
            set_labels=(
                f"{venn_datasets[0]}\n(n={len(s1)})",
                f"{venn_datasets[1]}\n(n={len(s2)})",
                f"{venn_datasets[2]}\n(n={len(s3)})",
            ),
            ax=ax
        )
        ax.set_title(
            f"Predicted RA Candidate DEGs Across Datasets\n"
            f"Confirmed in 2+ datasets: {len(confirmed_2plus)} proteins",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        venn_path = os.path.join(OUTPUT_DIR, "ML_DEG_venn_final.png")
        plt.savefig(venn_path, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"\nVenn diagram saved: {venn_path}")



# ==============================
# FINAL SUMMARY
# ==============================
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
for ds in dataset_names:
    df   = all_ml_deg_dfs[ds]
    up   = len(df[df["Direction"] == "Upregulated"])
    down = len(df[df["Direction"] == "Downregulated"])
    print(f"  {ds}: {len(df)} validated DEGs  (UP: {up}, DOWN: {down})")

if len(dataset_names) >= 2:
    print(f"\n  Confirmed in 2+ datasets: {len(confirmed_2plus)} genes")
    print(f"  Confirmed in all {len(dataset_names)} datasets: {len(confirmed_all)} genes")

print(f"\n  Outputs saved to: {OUTPUT_DIR}")
print(f"  Heatmaps: heatmap_novel_predicted_RA_proteins.png")
print(f"            heatmap_known_RA_seed_proteins.png")
print("DONE")