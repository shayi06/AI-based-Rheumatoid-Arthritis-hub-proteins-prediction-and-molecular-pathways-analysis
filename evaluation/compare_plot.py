import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc,
                             precision_recall_curve,
                             average_precision_score,
                             roc_auc_score)

# ===== PATHS =====
excel_path         = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\MIL\protein_predictions_mean_pool_cv.xlsx"
algo_scores_path   = r"D:\abi\abi\pythonProject\Abi 2025\Research\network\evaluation\algo_scores.json"
original_data_path = r"D:\abi\abi\pythonProject\Abi 2025\Research\network\network\all_scores.json"
save_dir           = os.path.dirname(excel_path)

NETWORK_ALGOS = ['MV', 'Hishi', 'RWR', 'FF', 'Ensemble']

MIL_MODELS = {
    "LogisticRegression" : "Logi",
    "SVM_RBF"            : "SVM_",
    "RandomForest"       : "Rand",
    "GradientBoosting"   : "Grad",
    "MLP"                : "MLP",
}

COLORS = {
    "LogisticRegression" : "#E91E63",
    "SVM_RBF"            : "#F44336",
    "RandomForest"       : "#9C27B0",
    "GradientBoosting"   : "#FF5722",
    "MLP"                : "#3F51B5",
    "MV"                 : "#2196F3",
    "Hishi"              : "#FF9800",
    "RWR"                : "#4CAF50",
    "FF"                 : "#00BCD4",
    "Ensemble"           : "#795548",
}

LINESTYLES = {
    "LogisticRegression" : {"lw": 2.5, "ls": "-"},
    "SVM_RBF"            : {"lw": 2.5, "ls": "-"},
    "RandomForest"       : {"lw": 3.0, "ls": "-"},
    "GradientBoosting"   : {"lw": 2.5, "ls": "-"},
    "MLP"                : {"lw": 2.0, "ls": "--"},
    "MV"                 : {"lw": 2.0, "ls": "--"},
    "Hishi"              : {"lw": 2.0, "ls": "--"},
    "RWR"                : {"lw": 2.0, "ls": "--"},
    "FF"                 : {"lw": 2.0, "ls": "--"},
    "Ensemble"           : {"lw": 2.5, "ls": "-."},
}

# =====================================================
# 1. LOAD MIL RESULTS
# =====================================================
print("🔹 Loading MIL Excel...")
df = pd.read_excel(excel_path)
print(f"  Columns: {list(df.columns[:8])}...")

mil_results = {}

for model_name, prefix in MIL_MODELS.items():
    prob_col = f"{prefix}_RHEUMATOID_Prob"
    true_col = "True_RHEUMATOID"

    if prob_col not in df.columns:
        print(f"⚠️  Missing: {prob_col}")
        continue

    sub    = df[[prob_col, true_col]].dropna()
    y_true = sub[true_col].values.astype(int)
    y_prob = sub[prob_col].values.astype(float)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap            = average_precision_score(y_true, y_prob)
    baseline      = y_true.sum() / len(y_true)

    mil_results[model_name] = {
        "fpr"     : fpr,
        "tpr"     : tpr,
        "roc_auc" : roc_auc,
        "prec"    : prec,
        "rec"     : rec,
        "ap"      : ap,
        "baseline": baseline
    }
    print(f"  {model_name:<22} ROC-AUC={roc_auc:.4f}  AP={ap:.4f}")

# =====================================================
# 2. LOAD NETWORK RESULTS
# =====================================================
print("\n🔹 Loading network algorithm scores...")

# algo_scores.json = global (not per fold)
with open(algo_scores_path, "r") as f:
    algo_data = json.load(f)

# all_scores.json = per fold info
with open(original_data_path, "r") as f:
    original_data = json.load(f)

network_results = {}

for algo in NETWORK_ALGOS:
    all_true   = []
    all_scores = []

    for fold in range(1, 11):
        fold_str   = str(fold)

        # test seeds = positives
        test_seeds = original_data[fold_str]['test_seeds']

        # all proteins in this fold
        all_prots  = list(original_data[fold_str]['MV'].keys())

        # non-seeds = negatives
        ns = list(set(all_prots) - set(test_seeds))

        # positives → label=1
        for p in test_seeds:
            all_true.append(1)
            if algo == 'Hishi':
                all_scores.append(algo_data['Hishi'].get(p, 0))
            else:
                all_scores.append(algo_data[algo].get(p, 0))

        # negatives → label=0
        for n in ns:
            all_true.append(0)
            if algo == 'Hishi':
                all_scores.append(algo_data['Hishi'].get(n, 0))
            else:
                all_scores.append(algo_data[algo].get(n, 0))

    true_arr  = np.array(all_true)
    score_arr = np.array(all_scores)

    fpr, tpr, _  = roc_curve(true_arr, score_arr)
    roc_auc      = roc_auc_score(true_arr, score_arr)
    prec, rec, _ = precision_recall_curve(true_arr, score_arr)
    ap           = auc(rec, prec)
    baseline     = true_arr.sum() / len(true_arr)

    network_results[algo] = {
        "fpr"     : fpr,
        "tpr"     : tpr,
        "roc_auc" : roc_auc,
        "prec"    : prec,
        "rec"     : rec,
        "ap"      : ap,
        "baseline": baseline
    }
    print(f"  {algo:<12} ROC-AUC={roc_auc:.4f}  AP={ap:.4f}")

# =====================================================
# 3. PLOT
# =====================================================
print("\n🔹 Plotting...")

fig, axes = plt.subplots(1, 2, figsize=(20, 7))
fig.suptitle(
    "ROC & Precision-Recall Curves\n"
    "MIL Models vs Network Algorithms (RHEUMATOID ARTHRITIS)",
    fontsize=14, fontweight='bold'
)

def get_label(model_name, metric, value):
    if model_name == "MLP":
        return f"DirectMLP-baseline  ({metric}={value:.4f})"
    return f"MIL-{model_name}  ({metric}={value:.4f})"

# ── LEFT: ROC ──
ax1 = axes[0]
ax1.plot([0, 1], [0, 1],
         color='gray', linestyle=':',
         linewidth=1.5, label='Random (AUC=0.50)')

for model_name in MIL_MODELS:
    if model_name not in mil_results:
        continue
    r     = mil_results[model_name]
    label = get_label(model_name, "AUC", r['roc_auc'])
    ax1.plot(r["fpr"], r["tpr"],
             color=COLORS[model_name],
             linewidth=LINESTYLES[model_name]["lw"],
             linestyle=LINESTYLES[model_name]["ls"],
             label=label)

for algo in NETWORK_ALGOS:
    r = network_results[algo]
    ax1.plot(r["fpr"], r["tpr"],
             color=COLORS[algo],
             linewidth=LINESTYLES[algo]["lw"],
             linestyle=LINESTYLES[algo]["ls"],
             label=f"Net-{algo}  (AUC={r['roc_auc']:.4f})")

ax1.set_title("ROC Curve", fontsize=13, fontweight='bold')
ax1.set_xlabel("False Positive Rate", fontsize=12)
ax1.set_ylabel("True Positive Rate", fontsize=12)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.02)
ax1.legend(fontsize=12, loc='lower right')
ax1.grid(True, alpha=0.3)

# ── RIGHT: PR ──
ax2 = axes[1]

for model_name in MIL_MODELS:
    if model_name not in mil_results:
        continue
    r     = mil_results[model_name]
    label = get_label(model_name, "AP", r['ap'])
    ax2.plot(r["rec"], r["prec"],
             color=COLORS[model_name],
             linewidth=LINESTYLES[model_name]["lw"],
             linestyle=LINESTYLES[model_name]["ls"],
             label=label)
    ax2.axhline(y=r["baseline"],
                color=COLORS[model_name],
                linestyle=':', linewidth=1.0, alpha=0.3)

for algo in NETWORK_ALGOS:
    r = network_results[algo]
    ax2.plot(r["rec"], r["prec"],
             color=COLORS[algo],
             linewidth=LINESTYLES[algo]["lw"],
             linestyle=LINESTYLES[algo]["ls"],
             label=f"Net-{algo}  (AP={r['ap']:.4f})")
    ax2.axhline(y=r["baseline"],
                color=COLORS[algo],
                linestyle=':', linewidth=1.0, alpha=0.3)

ax2.set_title("Precision-Recall Curve", fontsize=13, fontweight='bold')
ax2.set_xlabel("Recall", fontsize=12)
ax2.set_ylabel("Precision", fontsize=12)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1.02)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.text(0.01, 0.03,
         "Dotted horizontal = random baseline per method",
         fontsize=10, color='gray',
         transform=ax2.transAxes)

fig.text(0.5, 0.01,
         "Solid lines = MIL classifiers  |  "
         "Dashed lines = Network algorithms  |  "
         "MLP = DirectMLP baseline (raw 456-dim)",
         ha='center', fontsize=10, color='gray')

plt.tight_layout(rect=[0, 0.03, 1, 1])
plot_path = os.path.join(save_dir, "merged_roc_pr_curves.png")
plt.savefig(plot_path, dpi=500, bbox_inches='tight')
plt.show()
print(f"\n✅ Saved: {plot_path}")

# =====================================================
# 4. FINAL SUMMARY
# =====================================================
print("\n" + "="*55)
print("FINAL SUMMARY — RHEUMATOID ARTHRITIS")
print("="*55)
print(f"\n{'Method':<32} {'ROC-AUC':>8} {'AP':>8}")
print("─"*52)
print("MIL Models:")
for model_name in MIL_MODELS:
    if model_name not in mil_results:
        continue
    r    = mil_results[model_name]
    name = "DirectMLP-baseline" if model_name == "MLP" \
           else f"MIL-{model_name}"
    print(f"  {name:<30} {r['roc_auc']:>8.4f} {r['ap']:>8.4f}")

print("─"*52)
print("Network Algorithms:")
for algo in NETWORK_ALGOS:
    r = network_results[algo]
    print(f"  {'Net-'+algo:<30} {r['roc_auc']:>8.4f} {r['ap']:>8.4f}")