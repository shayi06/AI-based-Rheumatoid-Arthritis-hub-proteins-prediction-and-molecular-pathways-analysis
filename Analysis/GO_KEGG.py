import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from gprofiler import GProfiler

# ===== FILE PATHS =====
CONFIRMED_FILE = r"D:\abi\abi\pythonProject\Abi 2025\Research\network\seed_retrieving\ra_133_proteins.xlsx"
OUTPUT_DIR     = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\DEG analysis\pathway_analysis\known"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔹 Loading ML Novel proteins...")
df        = pd.read_excel(CONFIRMED_FILE)
gene_list = df["Gene_x"].dropna().tolist()

print(f"Total confirmed proteins: {len(gene_list)}")
print(f"Genes: {gene_list}")

# ===== 2. RUN GO + KEGG ENRICHMENT =====
print("\n🔹 Running GO and KEGG enrichment via gProfiler...")

gp      = GProfiler(return_dataframe=True)
results = gp.profile(
    organism                      = "hsapiens",
    query                         = gene_list,
    sources                       = ["GO:BP", "GO:MF", "GO:CC", "KEGG"],
    significance_threshold_method = "fdr",
    user_threshold                = 0.05,
    no_evidences                  = False
)

print(f"Total enriched terms: {len(results)}")

if results.empty:
    print("⚠️ No significant enrichment found.")
else:
    # Clean columns
    results = results[[
        "source", "native", "name",
        "p_value", "term_size",
        "intersection_size", "recall",
        "precision", "intersections"
    ]].copy()

    results["-log10(p)"] = -np.log10(results["p_value"].clip(lower=1e-300))
    results = results.sort_values("-log10(p)", ascending=False).reset_index(drop=True)

    # Split by source
    go_bp = results[results["source"] == "GO:BP"].head(20)
    go_mf = results[results["source"] == "GO:MF"].head(20)
    go_cc = results[results["source"] == "GO:CC"].head(20)
    kegg  = results[results["source"] == "KEGG"].head(20)

    print(f"GO:BP terms: {len(go_bp)}")
    print(f"GO:MF terms: {len(go_mf)}")
    print(f"GO:CC terms: {len(go_cc)}")
    print(f"KEGG terms:  {len(kegg)}")

    # ===== 3. SAVE =====
    out_excel = os.path.join(OUTPUT_DIR, "ML_pathway_enrichment.xlsx")
    with pd.ExcelWriter(out_excel) as writer:
        results.to_excel(writer, sheet_name="All_Results", index=False)
        go_bp.to_excel(  writer, sheet_name="GO_BP",       index=False)
        go_mf.to_excel(  writer, sheet_name="GO_MF",       index=False)
        go_cc.to_excel(  writer, sheet_name="GO_CC",       index=False)
        kegg.to_excel(   writer, sheet_name="KEGG",        index=False)
    print(f"\n✅ Results saved to: {out_excel}")

    # ===== 4. BUBBLE PLOT =====
    def bubble_plot(df_plot, title, filename, color):
        if df_plot.empty:
            print(f"⚠️ No data for {title}")
            return
        df_plot = df_plot.head(15).copy()
        df_plot["name_short"] = df_plot["name"].str[:55]
        df_plot = df_plot.sort_values("-log10(p)", ascending=True)

        fig, ax = plt.subplots(figsize=(11, 8))
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FFFFFF")

        scatter = ax.scatter(
            df_plot["-log10(p)"],
            range(len(df_plot)),
            s          = df_plot["intersection_size"] * 25,
            c          = df_plot["-log10(p)"],
            cmap       = color,
            alpha      = 0.85,
            edgecolors = "grey",
            linewidths = 0.5
        )

        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot["name_short"], fontsize=12)
        ax.set_xlabel("-log10(adjusted p-value)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("-log10(p-value)", fontsize=13)

        sizes   = [1, 5, 10, 20]
        handles = [plt.scatter([], [], s=s*25, color="gray",
                               alpha=0.6, label=str(s)) for s in sizes]
        ax.legend(handles=handles, title="Gene count",
                  loc="lower right", fontsize=12, title_fontsize=12)

        ax.axvline(x=-np.log10(0.05), color="red",
                   linestyle="--", alpha=0.5, lw=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close()
        print(f"✅ Plot saved: {save_path}")

    # ===== 5. BAR PLOT =====
    def bar_plot(df_plot, title, filename):
        if df_plot.empty:
            print(f"⚠️ No data for {title}")
            return
        df_plot = df_plot.head(15).copy()
        df_plot["name_short"] = df_plot["name"].str[:55]
        df_plot = df_plot.sort_values("-log10(p)", ascending=True)

        fig, ax = plt.subplots(figsize=(11, 7))
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FFFFFF")

        bars = ax.barh(
            df_plot["name_short"],
            df_plot["-log10(p)"],
            alpha=0.85, edgecolor="white", height=0.7
        )

        norm = plt.Normalize(df_plot["-log10(p)"].min(),
                             df_plot["-log10(p)"].max())
        cmap = plt.colormaps["YlOrRd"]
        for bar, val in zip(bars, df_plot["-log10(p)"]):
            bar.set_color(cmap(norm(val)))

        ax.axvline(x=-np.log10(0.05), color="red",
                   linestyle="--", alpha=0.6, lw=1.5, label="p=0.05")
        ax.set_xlabel("-log10(adjusted p-value)", fontsize=14)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.legend(fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="y", labelsize=14)

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close()
        print(f"✅ Plot saved: {save_path}")

    # ===== GENERATE PLOTS =====
    bubble_plot(go_bp, "GO Biological Process — ML Confirmed RA Proteins",
                "ML_GO_BP_bubble.png", "Blues")
    bubble_plot(go_mf, "GO Molecular Function — ML Confirmed RA Proteins",
                "ML_GO_MF_bubble.png", "Greens")
    bubble_plot(go_cc, "GO Cellular Component — ML Confirmed RA Proteins",
                "ML_GO_CC_bubble.png", "Purples")
    bar_plot(kegg,     "KEGG Pathways — ML Confirmed RA Proteins",
                "ML_KEGG_bar.png")

    # ===== SUMMARY =====
    print(f"\n============ PATHWAY SUMMARY ============")
    print(f"Proteins analyzed: {len(gene_list)}")
    print(f"\nTop 5 GO:BP terms:")
    if not go_bp.empty:
        print(go_bp[["name","p_value","intersection_size"]].head(5).to_string(index=False))
    print(f"\nTop 5 KEGG pathways:")
    if not kegg.empty:
        print(kegg[["name","p_value","intersection_size"]].head(5).to_string(index=False))