import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score

# ===== CONFIG =====
EXCEL_PATH = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\MIL\protein_predictions_mean_pool_cv.xlsx"
SAVE_DIR   = os.path.dirname(EXCEL_PATH)

DISEASE_GROUPS = ["DIABETES", "CARDIOVASCULAR", "RHEUMATOID", "OBESITY"]
THRESHOLDS     = np.arange(0.05, 0.96, 0.05)

COLORS = {
    "DIABETES":       "#2196F3",
    "CARDIOVASCULAR": "#F44336",
    "RHEUMATOID":     "#4CAF50",
    "OBESITY":        "#FF9800"
}

# ===== LOAD =====
print("Loading Excel...")
df = pd.read_excel(EXCEL_PATH)

# ===== HELPERS =====
def compute_metrics(y_true, y_prob, thresholds):
    precision, recall, f1 = [], [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precision.append(precision_score(y_true, y_pred, zero_division=0))
        recall.append(recall_score(y_true, y_pred, zero_division=0))
        f1.append(f1_score(y_true, y_pred, zero_division=0))
    return np.array(precision), np.array(recall), np.array(f1)

def max_distance_elbow(x_arr, y_arr, ref_arr):
    x1, y1 = x_arr[0],  y_arr[0]
    x2, y2 = x_arr[-1], y_arr[-1]
    dists = []
    for i in range(len(x_arr)):
        x0, y0 = x_arr[i], y_arr[i]
        num = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        den = np.sqrt((y2-y1)**2 + (x2-x1)**2) + 1e-9
        dists.append(num/den)
    idx = np.argmax(dists)
    return idx, x_arr[idx], y_arr[idx], ref_arr[idx]

def plot_metric_vs_threshold(thresholds, results, metric, colors, title, filename, marker='o'):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    for disease in DISEASE_GROUPS:
        if disease not in results: continue
        r = results[disease]
        ax.plot(thresholds, r[metric], color=colors[disease], linewidth=2.5,
                marker=marker, markersize=4, label=disease)
        best_idx = np.argmax(r[metric])
        ax.scatter([thresholds[best_idx]], [r[metric][best_idx]],
                   color=colors[disease], s=150, edgecolors='black',
                   linewidths=1.2, marker='D')
        ax.annotate(f"T={thresholds[best_idx]:.2f}",
                    xy=(thresholds[best_idx], r[metric][best_idx]),
                    xytext=(5, 6), textcoords='offset points',
                    fontsize=9, color=colors[disease], fontweight='bold')
    ax.set_xlabel("Threshold"); ax.set_ylabel(metric.capitalize())
    ax.set_xlim(0.05, 0.95); ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0.1, 1.0, 0.1))
    ax.grid(True, alpha=0.3); ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, filename), dpi=500, bbox_inches='tight')
    print(f"Saved {filename}")
    return fig, ax

# ===== COMPUTE METRICS =====
results = {}
for disease in DISEASE_GROUPS:
    prob_col = f"Rand_{disease}_Prob"
    true_col = f"True_{disease}"
    if prob_col not in df.columns or true_col not in df.columns:
        print(f"Columns missing for {disease}"); continue
    sub    = df[[prob_col, true_col]].dropna()
    y_true = sub[true_col].values.astype(int)
    y_prob = sub[prob_col].values.astype(float)
    prec, rec, f1 = compute_metrics(y_true, y_prob, THRESHOLDS)
    results[disease] = {"precision": prec, "recall": rec, "f1": f1}
    print(f"{disease} metrics computed")

# ===== PLOT 1: Precision vs Threshold =====
plot_metric_vs_threshold(THRESHOLDS, results, "precision", COLORS,
                         "Precision vs Threshold — All Diseases",
                         "plot1_precision_vs_threshold.png", marker='s')

# ===== PLOT 2: F1 vs Threshold =====
plot_metric_vs_threshold(THRESHOLDS, results, "f1", COLORS,
                         "F1 vs Threshold — All Diseases",
                         "plot2_f1_vs_threshold.png", marker='o')

# ===== PLOT 3: RA Average Precision+F1 =====
ra      = results["RHEUMATOID"]
ra_avg  = (ra["precision"] + ra["f1"]) / 2
best_idx = np.argmax(ra_avg)
best_t   = THRESHOLDS[best_idx]

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(THRESHOLDS, ra["precision"], linestyle='--', marker='s', color="#4CAF50", alpha=0.75, label="Precision")
ax3.plot(THRESHOLDS, ra["f1"],        linestyle='--', marker='o', color="#2196F3", alpha=0.75, label="F1")
ax3.plot(THRESHOLDS, ra_avg,          marker='D',     color="#E91E63", linewidth=3, label="Average (Precision+F1)/2")
ax3.axvline(x=best_t, color='red', linestyle=':', linewidth=2, alpha=0.6)
ax3.annotate(f"T={best_t:.2f}\nPrec={ra['precision'][best_idx]:.2f}  F1={ra['f1'][best_idx]:.2f}",
             xy=(best_t, ra_avg[best_idx]), xytext=(20, -40), textcoords='offset points',
             fontsize=10, color='red', fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))
ax3.set_xlabel("Threshold"); ax3.set_ylabel("Score")
ax3.set_xlim(0.05, 0.95); ax3.set_ylim(0, 1.05)
ax3.set_xticks(np.arange(0.1, 1.0, 0.1))
ax3.grid(True, alpha=0.3); ax3.legend(fontsize=11)
plt.tight_layout()
fig3.savefig(os.path.join(SAVE_DIR, "plot3_RA_avg_precision_f1.png"), dpi=150, bbox_inches='tight')
print("Saved plot3_RA_avg_precision_f1.png")

# ===== PLOT 5: Global Threshold =====
global_prec = np.mean([results[d]["precision"] for d in DISEASE_GROUPS if d in results], axis=0)
best_idx    = np.argmax(global_prec)
best_t      = THRESHOLDS[best_idx]

fig5, ax5 = plt.subplots(figsize=(10, 6))
ax5.plot(THRESHOLDS, global_prec, marker='o', color="#673AB7", linewidth=3, label="Avg Precision")
ax5.axvline(best_t, color='red', linestyle=':', linewidth=2)
ax5.annotate(f"Optimal Global Threshold\nT={best_t:.2f}",
             xy=(best_t, global_prec[best_idx]), xytext=(20, -40), textcoords='offset points',
             fontsize=10, color='red', fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))
ax5.set_xlabel("Threshold"); ax5.set_ylabel("Average Precision")
ax5.set_xlim(0.05, 0.95); ax5.set_ylim(0, 1.05)
ax5.grid(True, alpha=0.3); ax5.legend()
plt.tight_layout()
fig5.savefig(os.path.join(SAVE_DIR, "plot5_global_threshold.png"), dpi=150, bbox_inches='tight')
print("Saved plot5_global_threshold.png")

# ===== SHARED: fine-grained PR data for plot 6a & 6b =====
ra_true = df["True_RHEUMATOID"].dropna().values.astype(int)
ra_prob = df["Rand_RHEUMATOID_Prob"].dropna().values.astype(float)

# sklearn gives 473 fine-grained points — use these for BOTH plots
prec_sk, rec_sk, thr_sk = precision_recall_curve(ra_true, ra_prob)
prec_sk = prec_sk[:-1]   # drop last dummy point
rec_sk  = rec_sk[:-1]
# thr_sk already has len = len(prec_sk) after drop

# clip to meaningful recall range to avoid degenerate edges
mask   = (rec_sk >= 0.05) & (rec_sk <= 0.98)
prec_m = prec_sk[mask]
rec_m  = rec_sk[mask]
thr_m  = thr_sk[mask]

# find elbow ONCE on fine-grained clipped curve
_, elbow_rec, elbow_prec, elbow_t_pr = max_distance_elbow(rec_m, prec_m, thr_m)

auc_pr   = average_precision_score(ra_true, ra_prob)
baseline = ra_true.mean()

print(f"\nElbow found at: T={elbow_t_pr:.3f}  Precision={elbow_prec:.3f}  Recall={elbow_rec:.3f}")

# ===== PLOT 6a: PR Curve (Precision vs Recall) with elbow dot =====
fig6a, ax6a = plt.subplots(figsize=(10, 6))
ax6a.fill_between(rec_sk, prec_sk, alpha=0.08, color="#4CAF50")
ax6a.plot(rec_sk, prec_sk, color="#4CAF50", linewidth=2.5,
          label=f"PR Curve (AUC-PR={auc_pr:.3f})")
ax6a.axhline(baseline, color='gray', linestyle='--', linewidth=1.2, alpha=0.6,
             label=f"Baseline (pos rate={baseline:.2f})")
ax6a.scatter([elbow_rec], [elbow_prec], color="#F44336", s=200,
             edgecolors='black', linewidths=1.2, zorder=5,
             label=f"Elbow  T={elbow_t_pr:.3f}")
ax6a.annotate(
    f"Elbow (Max Distance)\nThreshold = {elbow_t_pr:.3f}\nPrecision = {elbow_prec:.3f}\nRecall    = {elbow_rec:.3f}",
    xy=(elbow_rec, elbow_prec), xytext=(30, 10), textcoords='offset points',
    fontsize=10, color="#F44336", fontweight='bold',
    bbox=dict(facecolor='white', edgecolor='#F44336', boxstyle='round,pad=0.4', alpha=0.9),
    arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.8)
)
ax6a.set_xlabel("Recall", fontsize=12)
ax6a.set_ylabel("Precision", fontsize=12)
ax6a.set_title("RHEUMATOID — PR Curve with Elbow Threshold (Max Distance)",
               fontsize=12, fontweight='bold')
ax6a.set_xlim(0, 1.05); ax6a.set_ylim(0, 1.05)
ax6a.grid(True, alpha=0.3); ax6a.legend(fontsize=10)
plt.tight_layout()
fig6a.savefig(os.path.join(SAVE_DIR, "plot6a_RA_PR_curve_elbow.png"), dpi=150, bbox_inches='tight')
print("Saved plot6a_RA_PR_curve_elbow.png")

# ===== PLOT 6b: Precision & Recall vs Threshold — SAME fine-grained data =====
# FIX: use thr_sk (fine) as x-axis, NOT the coarse THRESHOLDS grid
# sort by threshold ascending for a clean line plot
sort_idx   = np.argsort(thr_sk)
thr_sorted  = thr_sk[sort_idx]
prec_sorted = prec_sk[sort_idx]
rec_sorted  = rec_sk[sort_idx]

# find exact index of elbow threshold in fine array
elbow_idx   = np.argmin(np.abs(thr_sorted - elbow_t_pr))
elbow_t_exact   = thr_sorted[elbow_idx]
elbow_prec_exact = prec_sorted[elbow_idx]
elbow_rec_exact  = rec_sorted[elbow_idx]

fig6b, ax6b = plt.subplots(figsize=(10, 6))
ax6b.plot(thr_sorted, prec_sorted, label="Precision", color="#4CAF50", linewidth=2.5)
ax6b.plot(thr_sorted, rec_sorted,  label="Recall",    color="#F44336", linewidth=2.5)
ax6b.axvline(elbow_t_exact, color='red', linestyle=':', linewidth=2.2)
ax6b.scatter([elbow_t_exact], [elbow_prec_exact], color='red', s=180,
             edgecolors='black', linewidths=1.2, zorder=5)
ax6b.scatter([elbow_t_exact], [elbow_rec_exact], color='red', s=180,
             edgecolors='black', linewidths=1.2, zorder=5)
ax6b.annotate(
    f"Elbow Threshold\nT={elbow_t_exact:.3f}\nPrec={elbow_prec_exact:.3f}\nRec={elbow_rec_exact:.3f}",
    xy=(elbow_t_exact, (elbow_prec_exact + elbow_rec_exact) / 2),
    xytext=(20, 0), textcoords='offset points',
    fontsize=10, color='red', fontweight='bold',
    bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.4', alpha=0.9),
    arrowprops=dict(arrowstyle='->', color='red', lw=1.8)
)
ax6b.set_xlabel("Threshold", fontsize=12)
ax6b.set_ylabel("Score", fontsize=12)
ax6b.set_title("RHEUMATOID — Precision & Recall vs Threshold with Elbow",
               fontsize=12, fontweight='bold')
ax6b.set_xlim(0.05, 0.95); ax6b.set_ylim(0, 1.05)
ax6b.set_xticks(np.arange(0.1, 1.0, 0.1))
ax6b.grid(True, alpha=0.3); ax6b.legend(fontsize=11)
plt.tight_layout()
fig6b.savefig(os.path.join(SAVE_DIR, "plot6b_RA_threshold_vs_PR_elbow.png"), dpi=150, bbox_inches='tight')
print("Saved plot6b_RA_threshold_vs_PR_elbow.png")

plt.show()
print("\nAll plots saved!")
print(f"\nFinal Elbow Threshold: T={elbow_t_exact:.3f}")
print(f"  Precision = {elbow_prec_exact:.3f}")
print(f"  Recall    = {elbow_rec_exact:.3f}")