import io
import sys
import os
import json
import base64
import contextlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Script that runs Section 14, captures output and image, and embeds them into the notebook
nb_path = r"C:\adavtive_dataset\02_cold_start_ml_and_shap_analysis.ipynb"

code_source = [
    "# ==============================================================================\n",
    "# SECTION 14: STATISTICAL CONFIDENCE INTERVALS (95% CI)\n",
    "# 1. Non-Parametric Bootstrap (B=2,000 iterations on N=3,580 unseen BB-MAS taps)\n",
    "# 2. Closed-Form Analytical Wilson Score 95% CI for Accuracy\n",
    "# 3. 10-Seed Stability 95% Confidence Intervals (Student's t-distribution df=9)\n",
    "# 4. Publication-Quality Figure (fig_confidence_intervals_evaluation.png)\n",
    "# ==============================================================================\n",
    "\n",
    "import os\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "import joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import scipy.stats as stats\n",
    "import matplotlib\n",
    "if matplotlib.get_backend().lower() not in ['nbagg', 'inline', 'module://matplotlib_inline.backend_inline']:\n",
    "    matplotlib.use('Agg')\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score\n",
    "\n",
    "try:\n",
    "    base_dir = os.path.dirname(os.path.abspath(__file__))\n",
    "except NameError:\n",
    "    base_dir = os.getcwd()\n",
    "if not os.path.exists(os.path.join(base_dir, 'processed', 'bbmas_test_ready_dataset.csv')):\n",
    "    base_dir = r'C:\\adavtive_dataset'\n",
    "test_path = os.path.join(base_dir, \"processed\", \"bbmas_test_ready_dataset.csv\")\n",
    "scaler_path = os.path.join(base_dir, \"saved_models\", \"robust_feature_scaler.pkl\")\n",
    "model_path = os.path.join(base_dir, \"saved_models\", \"HistGradientBoosting.pkl\")\n",
    "figures_dir = os.path.join(base_dir, \"paper_figures\", \"cold_start_shap\")\n",
    "outputs_dir = os.path.join(base_dir, \"model_outputs\")\n",
    "os.makedirs(figures_dir, exist_ok=True)\n",
    "os.makedirs(outputs_dir, exist_ok=True)\n",
    "\n",
    "df_test = pd.read_csv(test_path)\n",
    "FEATURE_COLS = [\n",
    "    'context_is_walking', 'touch_duration_mean_ms', 'swipe_speed_mean_pxps',\n",
    "    'swipe_distance_mean_px', 'typing_speed_kpm', 'backspace_rate',\n",
    "    'contact_size_mean', 'touch_x_mean', 'touch_y_mean', 'tap_accuracy',\n",
    "    'miss_rate', 'reaction_time_mean_ms', 'reaction_time_norm'\n",
    "]\n",
    "TARGET_COL = 'target_button_size'\n",
    "CLASS_NAMES = ['Small (40dp)', 'Default (48dp)', 'Large (56dp)', 'Extra Large (64dp)']\n",
    "\n",
    "X_test = df_test[FEATURE_COLS].copy()\n",
    "y_test = df_test[TARGET_COL].copy().astype(int)\n",
    "\n",
    "scaler = joblib.load(scaler_path)\n",
    "model = joblib.load(model_path)\n",
    "\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "y_pred = model.predict(X_test_scaled)\n",
    "\n",
    "# Asymmetric Cost Matrix from Section 13\n",
    "COST_MATRIX = np.array([\n",
    "    [0, 1, 3, 6],\n",
    "    [2, 0, 1, 3],\n",
    "    [4, 2, 0, 1],\n",
    "    [8, 4, 2, 0]\n",
    "])\n",
    "\n",
    "def compute_risk(y_true, y_p):\n",
    "    return np.mean([COST_MATRIX[t, p] for t, p in zip(y_true, y_p)])\n",
    "\n",
    "# 1. Analytical Wilson Score 95% Confidence Interval for Accuracy\n",
    "N = len(y_test)\n",
    "p_hat = accuracy_score(y_test, y_pred)\n",
    "z = stats.norm.ppf(0.975)\n",
    "wilson_denom = 1 + z**2 / N\n",
    "wilson_center = (p_hat + z**2 / (2 * N)) / wilson_denom\n",
    "wilson_margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * N)) / N) / wilson_denom\n",
    "wilson_low = (wilson_center - wilson_margin) * 100.0\n",
    "wilson_high = (wilson_center + wilson_margin) * 100.0\n",
    "\n",
    "# 2. Non-Parametric Bootstrap (B = 2,000 resamples)\n",
    "np.random.seed(42)\n",
    "B = 2000\n",
    "boot_acc, boot_f1_macro, boot_f1_weighted, boot_risk = [], [], [], []\n",
    "boot_acc_sit, boot_acc_walk = [], []\n",
    "\n",
    "sit_mask = (df_test['context_is_walking'] == 0).values\n",
    "walk_mask = (df_test['context_is_walking'] == 1).values\n",
    "\n",
    "for _ in range(B):\n",
    "    idx = np.random.choice(N, size=N, replace=True)\n",
    "    y_t_b = y_test.iloc[idx].values\n",
    "    y_p_b = y_pred[idx]\n",
    "    \n",
    "    boot_acc.append(accuracy_score(y_t_b, y_p_b) * 100.0)\n",
    "    boot_f1_macro.append(f1_score(y_t_b, y_p_b, average='macro', zero_division=0) * 100.0)\n",
    "    boot_f1_weighted.append(f1_score(y_t_b, y_p_b, average='weighted', zero_division=0) * 100.0)\n",
    "    boot_risk.append(compute_risk(y_t_b, y_p_b))\n",
    "    \n",
    "    sit_idx = idx[sit_mask[idx]]\n",
    "    walk_idx = idx[walk_mask[idx]]\n",
    "    if len(sit_idx) > 0:\n",
    "        boot_acc_sit.append(accuracy_score(y_test.iloc[sit_idx], y_pred[sit_idx]) * 100.0)\n",
    "    if len(walk_idx) > 0:\n",
    "        boot_acc_walk.append(accuracy_score(y_test.iloc[walk_idx], y_pred[walk_idx]) * 100.0)\n",
    "\n",
    "def get_ci(arr):\n",
    "    return np.mean(arr), np.percentile(arr, 2.5), np.percentile(arr, 97.5), np.std(arr)\n",
    "\n",
    "m_acc, low_acc, high_acc, std_acc = get_ci(boot_acc)\n",
    "m_f1m, low_f1m, high_f1m, std_f1m = get_ci(boot_f1_macro)\n",
    "m_f1w, low_f1w, high_f1w, std_f1w = get_ci(boot_f1_weighted)\n",
    "m_risk, low_risk, high_risk, std_risk = get_ci(boot_risk)\n",
    "m_sit, low_sit, high_sit, std_sit = get_ci(boot_acc_sit)\n",
    "m_walk, low_walk, high_walk, std_walk = get_ci(boot_acc_walk)\n",
    "\n",
    "# Per-Class Bootstrap Metrics\n",
    "per_class_results = []\n",
    "for c in range(4):\n",
    "    boot_prec, boot_rec, boot_f1 = [], [], []\n",
    "    for _ in range(B):\n",
    "        idx = np.random.choice(N, size=N, replace=True)\n",
    "        y_t_b = (y_test.iloc[idx] == c).astype(int)\n",
    "        y_p_b = (y_pred[idx] == c).astype(int)\n",
    "        boot_prec.append(precision_score(y_t_b, y_p_b, zero_division=0) * 100.0)\n",
    "        boot_rec.append(recall_score(y_t_b, y_p_b, zero_division=0) * 100.0)\n",
    "        boot_f1.append(f1_score(y_t_b, y_p_b, zero_division=0) * 100.0)\n",
    "    \n",
    "    mp, lp, hp, _ = get_ci(boot_prec)\n",
    "    mr, lr, hr, _ = get_ci(boot_rec)\n",
    "    mf, lf, hf, _ = get_ci(boot_f1)\n",
    "    per_class_results.append({\n",
    "        'Class ID': c,\n",
    "        'Class Name': CLASS_NAMES[c],\n",
    "        'Precision (%)': f\"{mp:.2f}% [{lp:.2f}%, {hp:.2f}%]\",\n",
    "        'Recall (%)': f\"{mr:.2f}% [{lr:.2f}%, {hr:.2f}%]\",\n",
    "        'F1-Score (%)': f\"{mf:.2f}% [{lf:.2f}%, {hf:.2f}%]\",\n",
    "        'Mean_F1': mf, 'Low_F1': lf, 'High_F1': hf\n",
    "    })\n",
    "\n",
    "df_per_class_ci = pd.DataFrame(per_class_results)\n",
    "df_per_class_ci.to_csv(os.path.join(outputs_dir, \"04_confidence_intervals_per_class.csv\"), index=False)\n",
    "\n",
    "# Master Summary Table\n",
    "ci_master_summary = pd.DataFrame([\n",
    "    {'Metric': 'Overall Test Accuracy (Bootstrap B=2000)', 'Mean': f\"{m_acc:.2f}%\", '95% CI Lower': f\"{low_acc:.2f}%\", '95% CI Upper': f\"{high_acc:.2f}%\", 'Std Dev': f\"±{std_acc:.2f}%\"},\n",
    "    {'Metric': 'Overall Test Accuracy (Wilson Score)', 'Mean': f\"{p_hat*100:.2f}%\", '95% CI Lower': f\"{wilson_low:.2f}%\", '95% CI Upper': f\"{wilson_high:.2f}%\", 'Std Dev': 'N/A'},\n",
    "    {'Metric': 'Macro F1-Score', 'Mean': f\"{m_f1m:.2f}%\", '95% CI Lower': f\"{low_f1m:.2f}%\", '95% CI Upper': f\"{high_f1m:.2f}%\", 'Std Dev': f\"±{std_f1m:.2f}%\"},\n",
    "    {'Metric': 'Weighted F1-Score', 'Mean': f\"{m_f1w:.2f}%\", '95% CI Lower': f\"{low_f1w:.2f}%\", '95% CI Upper': f\"{high_f1w:.2f}%\", 'Std Dev': f\"±{std_f1w:.2f}%\"},\n",
    "    {'Metric': 'Sitting Context Accuracy', 'Mean': f\"{m_sit:.2f}%\", '95% CI Lower': f\"{low_sit:.2f}%\", '95% CI Upper': f\"{high_sit:.2f}%\", 'Std Dev': f\"±{std_sit:.2f}%\"},\n",
    "    {'Metric': 'Walking Context Accuracy', 'Mean': f\"{m_walk:.2f}%\", '95% CI Lower': f\"{low_walk:.2f}%\", '95% CI Upper': f\"{high_walk:.2f}%\", 'Std Dev': f\"±{std_walk:.2f}%\"},\n",
    "    {'Metric': 'Interaction Risk (Cost per Tap)', 'Mean': f\"{m_risk:.4f}\", '95% CI Lower': f\"{low_risk:.4f}\", '95% CI Upper': f\"{high_risk:.4f}\", 'Std Dev': f\"±{std_risk:.4f}\"}\n",
    "])\n",
    "ci_master_summary.to_csv(os.path.join(outputs_dir, \"04_confidence_intervals_summary.csv\"), index=False)\n",
    "\n",
    "# 10-Seed Summary\n",
    "seed_ci_csv = os.path.join(outputs_dir, \"04_multiseed_10_seeds_ci_comparison.csv\")\n",
    "df_seed_ci = pd.read_csv(seed_ci_csv)\n",
    "\n",
    "print(\"=\" * 90)\n",
    "print(\"TABLE 14A: NON-PARAMETRIC BOOTSTRAP 95% CONFIDENCE INTERVALS (B = 2,000, N = 3,580)\")\n",
    "print(\"=\" * 90)\n",
    "print(ci_master_summary.to_string(index=False))\n",
    "\n",
    "print(\"\\n\" + \"=\" * 90)\n",
    "print(\"TABLE 14B: PER-CLASS ADAPTIVE BUTTON SIZE 95% CONFIDENCE INTERVALS\")\n",
    "print(\"=\" * 90)\n",
    "print(df_per_class_ci[['Class ID', 'Class Name', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']].to_string(index=False))\n",
    "\n",
    "print(\"\\n\" + \"=\" * 90)\n",
    "print(\"TABLE 14C: MULTI-SEED STABILITY 95% CONFIDENCE INTERVALS (10 INDEPENDENT SEEDS, df=9)\")\n",
    "print(\"=\" * 90)\n",
    "print(df_seed_ci[['Model', 'Formatted 95% CI', 'Std Dev (%)']].to_string(index=False))\n",
    "\n",
    "# Plot publication figure\n",
    "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))\n",
    "\n",
    "# Subplot 1: Bootstrap Accuracy Distribution\n",
    "ax1 = axes[0]\n",
    "sns.histplot(boot_acc, kde=True, color='#1f77b4', bins=30, ax=ax1, stat='density', alpha=0.6)\n",
    "ax1.axvline(m_acc, color='#004d80', linestyle='-', linewidth=2.5, label=f'Mean = {m_acc:.2f}%')\n",
    "ax1.axvline(low_acc, color='#d62728', linestyle='--', linewidth=2, label=f'2.5% CI = {low_acc:.2f}%')\n",
    "ax1.axvline(high_acc, color='#d62728', linestyle='--', linewidth=2, label=f'97.5% CI = {high_acc:.2f}%')\n",
    "ax1.set_title('(A) Non-Parametric Bootstrap Test Accuracy\\n(B = 2,000 resamples, N = 3,580 unseen taps)', fontsize=12, fontweight='bold', pad=10)\n",
    "ax1.set_xlabel('Test Accuracy (%)', fontsize=11, fontweight='bold')\n",
    "ax1.set_ylabel('Probability Density', fontsize=11, fontweight='bold')\n",
    "ax1.legend(loc='upper right', frameon=True, fontsize=10)\n",
    "\n",
    "# Subplot 2: Multi-Model 95% CI Error Bars (10 Seeds)\n",
    "ax2 = axes[1]\n",
    "y_positions = np.arange(len(df_seed_ci))\n",
    "models_ordered = df_seed_ci['Model'].tolist()\n",
    "means = df_seed_ci['Mean Accuracy (%)'].values\n",
    "lows = df_seed_ci['95% CI Lower (%)'].values\n",
    "highs = df_seed_ci['95% CI Upper (%)'].values\n",
    "colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#d62728']\n",
    "\n",
    "for idx, (m_val, l_val, h_val, col) in enumerate(zip(means, lows, highs, colors)):\n",
    "    ax2.errorbar(m_val, idx, xerr=[[m_val - l_val], [h_val - m_val]], fmt='o', color=col, ecolor=col, elinewidth=2.5, capsize=6, capthick=2, markersize=8)\n",
    "    ax2.text(m_val, idx + 0.22, f\"{m_val:.2f}% [{l_val:.2f}%, {h_val:.2f}%]\", ha='center', va='bottom', fontsize=9, fontweight='bold', color=col)\n",
    "\n",
    "ax2.set_yticks(y_positions)\n",
    "ax2.set_yticklabels(models_ordered, fontsize=11, fontweight='bold')\n",
    "ax2.set_xlabel('Test Accuracy with 95% Confidence Interval (%)', fontsize=11, fontweight='bold')\n",
    "ax2.set_title('(B) Multi-Seed Stability (10 Random Seeds)\\nStudent\\'s t-distribution 95% CI (df=9)', fontsize=12, fontweight='bold', pad=10)\n",
    "ax2.set_xlim(85, 93)\n",
    "\n",
    "# Subplot 3: Per-Class F1-Scores with 95% CI\n",
    "ax3 = axes[2]\n",
    "c_means = df_per_class_ci['Mean_F1'].values\n",
    "c_lows = df_per_class_ci['Low_F1'].values\n",
    "c_highs = df_per_class_ci['High_F1'].values\n",
    "c_errs = [c_means - c_lows, c_highs - c_means]\n",
    "bar_colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']\n",
    "\n",
    "bars = ax3.bar(range(4), c_means, yerr=c_errs, capsize=6, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=1.2)\n",
    "for idx, (bar, m_val, l_val, h_val) in enumerate(zip(bars, c_means, c_lows, c_highs)):\n",
    "    ax3.text(bar.get_x() + bar.get_width() / 2, m_val / 2, f\"{m_val:.1f}%\\n[{l_val:.1f}%, {h_val:.1f}%]\", ha='center', va='center', fontsize=9.5, fontweight='bold', color='white')\n",
    "\n",
    "ax3.set_xticks(range(4))\n",
    "ax3.set_xticklabels(['Small\\n(40dp)', 'Default\\n(48dp)', 'Large\\n(56dp)', 'Extra Large\\n(64dp)'], fontsize=10, fontweight='bold')\n",
    "ax3.set_ylabel('F1-Score (%)', fontsize=11, fontweight='bold')\n",
    "ax3.set_title('(C) Per-Class Adaptation F1-Scores\\n(2,000 Bootstrap 95% CI)', fontsize=12, fontweight='bold', pad=10)\n",
    "ax3.set_ylim(0, 105)\n",
    "\n",
    "plt.tight_layout()\n",
    "fig_path = os.path.join(figures_dir, \"fig_confidence_intervals_evaluation.png\")\n",
    "plt.savefig(fig_path, dpi=300, bbox_inches='tight')\n",
    "plt.show()\n"
]

# Execute the code to capture stdout and figure
print("Executing Section 14 code to capture outputs...")
full_code_str = "".join(code_source)

stdout_buffer = io.StringIO()
with contextlib.redirect_stdout(stdout_buffer):
    exec_scope = {}
    exec(full_code_str, exec_scope)

captured_stdout = stdout_buffer.getvalue()

# Load the saved figure and convert to base64
fig_path = os.path.join(r"C:\adavtive_dataset\paper_figures\cold_start_shap", "fig_confidence_intervals_evaluation.png")
with open(fig_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode('utf-8')

# Construct outputs
outputs = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [line + "\n" for line in captured_stdout.splitlines()]
    },
    {
        "data": {
            "image/png": img_b64,
            "text/plain": ["<Figure size 1800x550 with 3 Axes>"]
        },
        "metadata": {},
        "output_type": "display_data"
    }
]

# Update notebook
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 33 is the code cell for Section 14
nb['cells'][33]['source'] = code_source
nb['cells'][33]['execution_count'] = 14
nb['cells'][33]['outputs'] = outputs

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Successfully embedded full executable code and outputs in Cell 33 of {nb_path}!")
