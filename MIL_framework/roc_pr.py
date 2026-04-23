import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc,
                             precision_recall_curve,
                             average_precision_score)

# ===== PATH =====
excel_path = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\MIL\protein_predictions_mean_pool_cv.xlsx"
save_dir   = os.path.dirname(excel_path)

DISEASE_GROUPS = ["DIABETES", "CARDIOVASCULAR", "RHEUMATOID", "OBESITY"]

COLORS = {
    "DIABETES"      : "#2196F3",
    "CARDIOVASCULAR": "#F44336",
    "RHEUMATOID"    : "#4CAF50",
    "OBESITY"       : "#FF9800"
}

# ===== LOAD =====
print("🔹 Loading Excel...")
df = pd.read_excel(excel_path)
print(f"Total proteins: {len(df)}")

prob_cols = [c for c in df.columns if "Prob" in c]
print(f"Available prob columns: {prob_cols}")

# ===================================================================
# COLLECT DATA — RandomForest probabilities (best RA model)
# ===================================================================
results = {}
print("\n📊 AUC & AP Summary — RandomForest:")
print(f"{'Disease':<18} {'ROC-AUC':>8} {'AP':>8}")
print("─"*38)

for disease in DISEASE_GROUPS:
    prob_col = f"Rand_{disease}_Prob"   # ← RF prefix!
    true_col = f"True_{disease}"

    if prob_col not in df.columns:
        print(f"⚠️  Missing: {prob_col}")
        continue

    sub    = df[[prob_col, true_col]].dropna()
    y_true = sub[true_col].values.astype(int)
    y_prob = sub[prob_col].values.astype(float)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap                   = average_precision_score(y_true, y_prob)

    baseline = y_true.sum() / len(y_true)

    results[disease] = {
        "fpr"      : fpr,
        "tpr"      : tpr,
        "roc_auc"  : roc_auc,
        "precision": precision,
        "recall"   : recall,
        "ap"       : ap,
        "baseline" : baseline,
    }

    print(f"{disease:<18} {roc_auc:>8.4f} {ap:>8.4f}")

# ===================================================================
# PLOT
# ===================================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(
    "ROC Curve & Precision-Recall Curve — All 4 Diseases\n"
    "RandomForest OOF Probabilities from 10-Fold CV",
    fontsize=15, fontweight='bold')

# ── LEFT: ROC Curve ──
ax1 = axes[0]
ax1.plot([0, 1], [0, 1],
         color='gray', linestyle='--',
         linewidth=1.5, label='Random (AUC=0.50)')

for disease in DISEASE_GROUPS:
    if disease not in results:
        continue
    r     = results[disease]
    color = COLORS[disease]
    ax1.plot(r["fpr"], r["tpr"],
             color=color, linewidth=2.5,
             label=f"{disease}  (AUC={r['roc_auc']:.4f})")

ax1.set_title("ROC Curve", fontsize=13, fontweight='bold')
ax1.set_xlabel("False Positive Rate", fontsize=12)
ax1.set_ylabel("True Positive Rate", fontsize=12)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.02)
ax1.legend(fontsize=10, loc='lower right')
ax1.grid(True, alpha=0.3)

# ── RIGHT: Precision-Recall Curve ──
ax2 = axes[1]

for disease in DISEASE_GROUPS:
    if disease not in results:
        continue
    r     = results[disease]
    color = COLORS[disease]
    ax2.plot(r["recall"], r["precision"],
             color=color, linewidth=2.5,
             label=f"{disease}  (AP={r['ap']:.4f})")
    ax2.axhline(y=r["baseline"],
                color=color, linestyle=':',
                linewidth=1.0, alpha=0.5)

ax2.set_title("Precision-Recall Curve", fontsize=13, fontweight='bold')
ax2.set_xlabel("Recall", fontsize=12)
ax2.set_ylabel("Precision", fontsize=12)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1.02)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.text(0.01, 0.03,
         "Dotted line = random baseline per disease",
         fontsize=8, color='gray',
         transform=ax2.transAxes)

plt.tight_layout()
plot_path = os.path.join(save_dir, "roc_pr_curves_rf.png")   # ← rf!
plt.savefig(plot_path, dpi=500, bbox_inches='tight')
plt.show()
print(f"\n✅ Plot saved: {plot_path}")