import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, average_precision_score

# ===== PATHS =====
excel_path         = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\MIL\protein_predictions_mean_pool_cv.xlsx"
algo_scores_path   = r"D:\abi\abi\pythonProject\Abi 2025\Research\network\evaluation\algo_scores.json"
original_data_path = r"D:\abi\abi\pythonProject\Abi 2025\Research\network\network\all_scores.json"

DISEASE_GROUPS = ["RHEUMATOID"]

MIL_MODELS = {
    "LogisticRegression" : "Logi",
    "SVM_RBF"            : "SVM_",
    "RandomForest"       : "Rand",
    "GradientBoosting"   : "Grad",
    "MLP"                : "MLP",
}

NETWORK_ALGOS = ['MV', 'Hishi', 'RWR', 'FF', 'Ensemble']

# ===== LOAD EXCEL =====
print("🔹 Loading Excel...")
df = pd.read_excel(excel_path)

# =====================================================
# 1. MIL MODELS
# =====================================================
print("\n🔹 Computing MIL model scores...")
mil_results = {}

for model_name, prefix in MIL_MODELS.items():
    mil_results[model_name] = {}
    for disease in DISEASE_GROUPS:

        prob_col = f"{prefix}_{disease}_Prob"
        pred_col = f"{prefix}_{disease}_Pred"
        true_col = f"True_{disease}"

        if prob_col not in df.columns:
            print(f"⚠️  Missing: {prob_col}")
            continue

        sub    = df[[prob_col, pred_col, true_col]].dropna()
        y_true = sub[true_col].values.astype(int)
        y_prob = sub[prob_col].values.astype(float)
        y_pred = sub[pred_col].values.astype(int)

        f1 = f1_score(y_true, y_pred, zero_division=0)
        ap = average_precision_score(y_true, y_prob)

        mil_results[model_name][disease] = {
            "f1": round(f1, 4),
            "ap": round(ap, 4)
        }
        type_label = "DirectMLP-baseline" \
                     if model_name == "MLP" else "MIL"
        print(f"  {model_name:<22} {disease:<18} "
              f"F1={f1:.4f}  AP={ap:.4f}  [{type_label}]")

# =====================================================
# 2. NETWORK ALGOS
# =====================================================
print("\n🔹 Computing network algorithm scores...")
with open(algo_scores_path, "r") as f:
    algo_data = json.load(f)
with open(original_data_path, "r") as f:
    original_data = json.load(f)

network_results = {}

for algo in NETWORK_ALGOS:
    all_true   = []
    all_scores = []

    # CORRECT way — same as your evaluation code
    for fold in range(1, 11):
        fold_str = str(fold)
        test_seeds = original_data[fold_str]['test_seeds']
        all_prots = list(original_data[fold_str]['MV'].keys())  # ← from all_scores.json
        ns = list(set(all_prots) - set(test_seeds))

        for p in test_seeds:
            all_true.append(1)
            all_scores.append(algo_data[algo].get(p, 0))  # ← score from algo_scores.json

        for n in ns:
            all_true.append(0)
            all_scores.append(algo_data[algo].get(n, 0))

    true_arr  = np.array(all_true)
    score_arr = np.array(all_scores)

    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.96, 0.05):
        pred = (score_arr >= t).astype(int)
        f1   = f1_score(true_arr, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    best_pred = (score_arr >= best_t).astype(int)
    f1        = f1_score(true_arr, best_pred, zero_division=0)
    ap        = average_precision_score(true_arr, score_arr)

    network_results[algo] = {
        "f1": round(f1, 4),
        "ap": round(ap, 4)
    }
    print(f"  {algo:<12} F1={f1:.4f}  AP={ap:.4f}  "
          f"(best_t={best_t:.2f})")

# =====================================================
# 3. PRINT COMPARISON TABLE
# =====================================================
print("\n" + "="*75)
print("FULL COMPARISON TABLE — MIL Models vs DirectMLP vs Network")
print("="*75)

for disease in DISEASE_GROUPS:
    print(f"\n── {disease} — MIL Pipeline ──")
    print(f"  {'Model':<22} {'F1':>8} {'AP':>8}  {'Type'}")
    print(f"  {'─'*55}")
    for model_name in MIL_MODELS:
        vals       = mil_results[model_name].get(disease, {})
        f1         = vals.get("f1", float('nan'))
        ap         = vals.get("ap", float('nan'))
        type_label = "DirectMLP-baseline" \
                     if model_name == "MLP" else "MIL"  # ← fixed!
        print(f"  {model_name:<22} {f1:>8.4f} {ap:>8.4f}  {type_label}")

print(f"\n── NETWORK ALGORITHMS ──")
print(f"  {'Model':<22} {'F1':>8} {'AP':>8}  {'Type'}")
print(f"  {'─'*55}")
for algo in NETWORK_ALGOS:
    vals = network_results[algo]
    print(f"  {algo:<22} {vals['f1']:>8.4f} "
          f"{vals['ap']:>8.4f}  Network")

# =====================================================
# 4. SAVE EXCEL
# =====================================================
rows = []

for model_name in MIL_MODELS:
    for disease in DISEASE_GROUPS:
        vals = mil_results[model_name].get(disease, {})
        rows.append({
            "Type"   : "DirectMLP-baseline" \
                       if model_name == "MLP" else "MIL",  # ← fixed!
            "Model"  : model_name,
            "Disease": disease,
            "F1"     : vals.get("f1", float('nan')),
            "AP"     : vals.get("ap", float('nan'))
        })

for algo in NETWORK_ALGOS:
    vals = network_results[algo]
    rows.append({
        "Type"   : "Network",
        "Model"  : algo,
        "Disease": "RHEUMATOID",
        "F1"     : vals["f1"],
        "AP"     : vals["ap"]
    })

df_out    = pd.DataFrame(rows)
save_path = excel_path.replace(".xlsx", "_model_comparison.xlsx")
df_out.to_excel(save_path, index=False)
print(f"\n✅ Saved: {save_path}")
print(f"Total rows: {len(df_out)}")

# =====================================================
# 5. BAR PLOT
# =====================================================
print("\n🔹 Plotting...")

all_models = list(MIL_MODELS.keys()) + NETWORK_ALGOS
all_f1     = [mil_results[m]["RHEUMATOID"]["f1"]
              for m in MIL_MODELS] + \
             [network_results[a]["f1"] for a in NETWORK_ALGOS]
all_ap     = [mil_results[m]["RHEUMATOID"]["ap"]
              for m in MIL_MODELS] + \
             [network_results[a]["ap"] for a in NETWORK_ALGOS]

# ← MLP shown as DirectMLP
display_names = ["LR", "SVM", "RF", "GB", "DirectMLP",
                 "MV", "Hishi", "RWR", "FF", "Ensemble"]

x     = np.arange(len(all_models))
width = 0.35

fig, ax = plt.subplots(figsize=(15, 6))

bars_f1 = ax.bar(x - width/2, all_f1,
                 width, label='F1 Score',
                 color='#2196F3', alpha=0.85,
                 edgecolor='black', linewidth=0.8)

bars_ap = ax.bar(x + width/2, all_ap,
                 width, label='Average Precision (AP)',
                 color='#FF9800', alpha=0.85,
                 edgecolor='black', linewidth=0.8)

for bar in bars_f1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2,
            h + 0.01, f'{h:.3f}',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold',
            color='#1565C0')

for bar in bars_ap:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2,
            h + 0.01, f'{h:.3f}',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold',
            color='#E65100')

# ← Three regions: MIL | DirectMLP | Network
ax.axvspan(-0.5, 3.5,
           alpha=0.06, color='#2196F3',
           label='MIL Pipeline')
ax.axvspan(3.5, 4.5,
           alpha=0.06, color='#3F51B5',
           label='DirectMLP Baseline')
ax.axvspan(4.5, len(all_models) - 0.5,
           alpha=0.06, color='#4CAF50',
           label='Network Methods')

# ← Two divider lines
ax.axvline(x=3.5, color='gray',
           linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(x=4.5, color='gray',
           linestyle='--', linewidth=1.5, alpha=0.7)

# ← Three region labels
ax.text(1.5, 0.97, 'MIL Pipeline',
        ha='center', transform=ax.get_xaxis_transform(),
        fontsize=10, color='#1565C0', fontweight='bold')
ax.text(4.0, 0.97, 'Baseline',
        ha='center', transform=ax.get_xaxis_transform(),
        fontsize=9, color='#3F51B5', fontweight='bold')
ax.text(7.0, 0.97, 'Network Methods',
        ha='center', transform=ax.get_xaxis_transform(),
        fontsize=10, color='#2E7D32', fontweight='bold')

ax.set_title("Model Comparison — F1 & AP\n"
             "MIL Pipeline vs DirectMLP Baseline vs "
             "Network Propagation (RHEUMATOID)",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(display_names, fontsize=10)
ax.set_ylim(0, 1.10)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plot_path = excel_path.replace(".xlsx", "_comparison_barplot.png")
plt.savefig(plot_path, dpi=500, bbox_inches='tight')
plt.show()
print(f"✅ Plot saved: {plot_path}")
