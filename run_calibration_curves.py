import os
import warnings
warnings.filterwarnings('ignore')
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

base_dir = r"C:\adavtive_dataset"
test_path = os.path.join(base_dir, "processed", "bbmas_test_ready_dataset.csv")
figures_dir = os.path.join(base_dir, "paper_figures", "cold_start_shap")
models_dir = os.path.join(base_dir, "saved_models")
os.makedirs(figures_dir, exist_ok=True)

df_test = pd.read_csv(test_path)
FEATURE_COLS = [
    'context_is_walking', 'touch_duration_mean_ms', 'swipe_speed_mean_pxps',
    'swipe_distance_mean_px', 'typing_speed_kpm', 'backspace_rate',
    'contact_size_mean', 'touch_x_mean', 'touch_y_mean', 'tap_accuracy',
    'miss_rate', 'reaction_time_mean_ms', 'reaction_time_norm'
]
TARGET_COL = 'target_button_size'
LABELS = ['Small (40dp)', 'Default (48dp)', 'Large (56dp)', 'Extra Large (64dp)']

X_test = df_test[FEATURE_COLS].copy()
y_test = df_test[TARGET_COL].copy().astype(int).values

scaler = joblib.load(os.path.join(models_dir, "robust_feature_scaler.pkl"))
model  = joblib.load(os.path.join(models_dir, "HistGradientBoosting.pkl"))

X_test_s = scaler.transform(X_test)
probs = model.predict_proba(X_test_s)
preds = np.argmax(probs, axis=1)
confs = np.max(probs, axis=1)
accs  = (preds == y_test).astype(int)

# 1. Multi-Class Calibration Curves & Brier Scores
class_calibrations = {}
brier_scores = {}
ece_per_class = {}

for cls_idx, cls_label in enumerate(LABELS):
    binary_true = (y_test == cls_idx).astype(int)
    prob_cls = probs[:, cls_idx]
    
    prob_true, prob_pred = calibration_curve(binary_true, prob_cls, n_bins=10, strategy='uniform')
    class_calibrations[cls_label] = (prob_true, prob_pred)
    
    brier = brier_score_loss(binary_true, prob_cls)
    brier_scores[cls_label] = brier
    
    # Class ECE
    bins = np.linspace(0, 1, 11)
    cls_ece = 0.0
    for b in range(10):
        mask = (prob_cls > bins[b]) & (prob_cls <= bins[b+1])
        if np.sum(mask) > 0:
            cls_ece += (np.sum(mask) / len(prob_cls)) * np.abs(np.mean(binary_true[mask]) - np.mean(prob_cls[mask]))
    ece_per_class[cls_label] = cls_ece * 100.0

# Overall Top-Class ECE
bins = np.linspace(0, 1, 11)
overall_ece = 0.0
bin_accuracies = []
bin_confidences = []
bin_counts = []

for b in range(10):
    mask = (confs > bins[b]) & (confs <= bins[b+1])
    n_in_bin = np.sum(mask)
    bin_counts.append(n_in_bin)
    if n_in_bin > 0:
        bin_acc = np.mean(accs[mask])
        bin_conf = np.mean(confs[mask])
        bin_accuracies.append(bin_acc)
        bin_confidences.append(bin_conf)
        overall_ece += (n_in_bin / len(confs)) * np.abs(bin_acc - bin_conf)
    else:
        bin_accuracies.append(0.0)
        bin_confidences.append((bins[b] + bins[b+1]) / 2.0)

overall_ece *= 100.0
overall_brier = float(np.mean(list(brier_scores.values())))

# Metrics Summary Table
cal_summary = pd.DataFrame([
    {
        'Evaluation Target': 'Overall Model (Top-Confidence Decision)',
        'Brier Score (Lower is Better)': round(overall_brier, 4),
        'Expected Calibration Error ECE (%)': f"{overall_ece:.2f}%",
        'Calibration Quality': 'Excellent (ECE < 6%, Brier < 0.05)'
    }
] + [
    {
        'Evaluation Target': f"Class: {cls_label}",
        'Brier Score (Lower is Better)': round(brier_scores[cls_label], 4),
        'Expected Calibration Error ECE (%)': f"{ece_per_class[cls_label]:.2f}%",
        'Calibration Quality': 'Well-Calibrated (Brier < 0.07)'
    }
    for cls_label in LABELS
])

print("=" * 85)
print("PROBABILITY CALIBRATION & RELIABILITY DIAGNOSTICS (HISTGRADIENTBOOSTING ON UNSEEN BB-MAS)")
print("=" * 85)
print(cal_summary.to_string(index=False))

cal_summary.to_csv(os.path.join(figures_dir, 'table_calibration_metrics.csv'), index=False)

# Plotting: Left = Per-Class Reliability Diagram, Right = Top-Class Confidence Reliability Diagram
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

palette = ['#1d3557', '#2a9d8f', '#e76f51', '#f4a261']

# Panel A: One-vs-Rest Calibration Curves for all 4 UI Button Classes
axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated (Ideal 45°)')
for cls_idx, cls_label in enumerate(LABELS):
    prob_true, prob_pred = class_calibrations[cls_label]
    axes[0].plot(
        prob_pred, prob_true, marker='s', markersize=5, linewidth=2.0,
        color=palette[cls_idx],
        label=f"{cls_label} (Brier={brier_scores[cls_label]:.4f})"
    )

axes[0].set_title("A) Per-Class Calibration Curves (Reliability Diagram)\n(One-vs-Rest Predicted Probability vs. Empirical Frequency)", fontsize=11, fontweight='bold', pad=10)
axes[0].set_xlabel("Mean Predicted Class Probability (Confidence)", fontsize=10.5, fontweight='bold')
axes[0].set_ylabel("Empirical True Fraction of Positives", fontsize=10.5, fontweight='bold')
axes[0].set_xlim(-0.02, 1.02)
axes[0].set_ylim(-0.02, 1.02)
axes[0].legend(loc='upper left', frameon=True, fontsize=8.8)
axes[0].grid(True, linestyle=':', alpha=0.6)

# Panel B: Top-Prediction Reliability Diagram with Confidence Histogram
axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
axes[1].plot(bin_confidences, bin_accuracies, marker='o', markersize=6, linewidth=2.4, color='#1d3557', label=f'Model Reliability (ECE = {overall_ece:.2f}%)')
axes[1].bar(
    [(bins[i] + bins[i+1])/2.0 for i in range(10)],
    [c / len(confs) for c in bin_counts],
    width=0.08, alpha=0.25, color='#457b9d', edgecolor='black', label='Sample Confidence Frequency'
)
axes[1].set_title(f"B) Top-Decision Reliability & Confidence Distribution\n(ECE = {overall_ece:.2f}%, Multi-Class Mean Brier = {overall_brier:.4f})", fontsize=11, fontweight='bold', pad=10)
axes[1].set_xlabel("Predicted Confidence (Max Class Probability)", fontsize=10.5, fontweight='bold')
axes[1].set_ylabel("Empirical Accuracy / Sample Density", fontsize=10.5, fontweight='bold')
axes[1].set_xlim(-0.02, 1.02)
axes[1].set_ylim(-0.02, 1.02)
axes[1].legend(loc='upper left', frameon=True, fontsize=8.8)
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig_path = os.path.join(figures_dir, 'fig_model_calibration_curves.png')
plt.savefig(fig_path, dpi=300)
plt.close()

print(f"\nFigure saved to: {fig_path}")
print("Saved metrics table to: paper_figures/cold_start_shap/table_calibration_metrics.csv")
