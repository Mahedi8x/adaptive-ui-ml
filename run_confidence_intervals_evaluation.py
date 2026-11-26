import os
import warnings
warnings.filterwarnings('ignore')
import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Paths
base_dir = r"C:\adavtive_dataset"
test_path = os.path.join(base_dir, "processed", "bbmas_test_ready_dataset.csv")
scaler_path = os.path.join(base_dir, "saved_models", "robust_feature_scaler.pkl")
model_path = os.path.join(base_dir, "saved_models", "HistGradientBoosting.pkl")
figures_dir = os.path.join(base_dir, "paper_figures", "cold_start_shap")
outputs_dir = os.path.join(base_dir, "model_outputs")
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

# Load data and artifacts using joblib
df_test = pd.read_csv(test_path)
scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

FEATURE_COLS = [
    'context_is_walking',
    'touch_duration_mean_ms',
    'swipe_speed_mean_pxps',
    'swipe_distance_mean_px',
    'typing_speed_kpm',
    'backspace_rate',
    'contact_size_mean',
    'touch_x_mean',
    'touch_y_mean',
    'tap_accuracy',
    'miss_rate',
    'reaction_time_mean_ms',
    'reaction_time_norm'
]
TARGET_COL = 'target_button_size'

X_test = df_test[FEATURE_COLS].copy()
y_test = df_test[TARGET_COL].copy().astype(int)
X_test_scaled = scaler.transform(X_test)

# Predict
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

# Asymmetric Cost Matrix from Section 13
COST_MATRIX = np.array([
    [0, 1, 3, 6],
    [2, 0, 1, 3],
    [4, 2, 0, 1],
    [8, 4, 2, 0]
])

def compute_risk(y_true, y_p):
    costs = [COST_MATRIX[t, p] for t, p in zip(y_true, y_p)]
    return np.mean(costs)

# 1. Wilson Score 95% Confidence Interval for Overall Accuracy
N = len(y_test)
p_hat = accuracy_score(y_test, y_pred)
z = stats.norm.ppf(0.975) # 1.95996
wilson_denom = 1 + z**2 / N
wilson_center = (p_hat + z**2 / (2 * N)) / wilson_denom
wilson_margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * N)) / N) / wilson_denom
wilson_low = (wilson_center - wilson_margin) * 100.0
wilson_high = (wilson_center + wilson_margin) * 100.0
print(f"Wilson Score 95% CI for Accuracy: {p_hat*100:.2f}% [{wilson_low:.2f}%, {wilson_high:.2f}%]")

# 2. Non-Parametric Bootstrap (B = 2,000 iterations)
np.random.seed(42)
B = 2000
boot_acc = []
boot_f1_macro = []
boot_f1_weighted = []
boot_risk = []
boot_acc_sit = []
boot_acc_walk = []

sit_mask = (df_test['context_is_walking'] == 0).values
walk_mask = (df_test['context_is_walking'] == 1).values

for _ in range(B):
    idx = np.random.choice(N, size=N, replace=True)
    y_t_b = y_test.iloc[idx].values
    y_p_b = y_pred[idx]
    
    # overall metrics
    boot_acc.append(accuracy_score(y_t_b, y_p_b) * 100.0)
    boot_f1_macro.append(f1_score(y_t_b, y_p_b, average='macro', zero_division=0) * 100.0)
    boot_f1_weighted.append(f1_score(y_t_b, y_p_b, average='weighted', zero_division=0) * 100.0)
    boot_risk.append(compute_risk(y_t_b, y_p_b))
    
    # context subsets in this bootstrap sample
    sit_idx = idx[sit_mask[idx]]
    walk_idx = idx[walk_mask[idx]]
    
    if len(sit_idx) > 0:
        boot_acc_sit.append(accuracy_score(y_test.iloc[sit_idx], y_pred[sit_idx]) * 100.0)
    if len(walk_idx) > 0:
        boot_acc_walk.append(accuracy_score(y_test.iloc[walk_idx], y_pred[walk_idx]) * 100.0)

def get_ci(arr):
    return np.mean(arr), np.percentile(arr, 2.5), np.percentile(arr, 97.5), np.std(arr)

m_acc, low_acc, high_acc, std_acc = get_ci(boot_acc)
m_f1m, low_f1m, high_f1m, std_f1m = get_ci(boot_f1_macro)
m_f1w, low_f1w, high_f1w, std_f1w = get_ci(boot_f1_weighted)
m_risk, low_risk, high_risk, std_risk = get_ci(boot_risk)
m_sit, low_sit, high_sit, std_sit = get_ci(boot_acc_sit)
m_walk, low_walk, high_walk, std_walk = get_ci(boot_acc_walk)

print("\n--- 2,000 Bootstrap 95% Confidence Intervals (Unseen BB-MAS Test Set) ---")
print(f"Overall Accuracy:  {m_acc:.2f}% [95% CI: {low_acc:.2f}% - {high_acc:.2f}%] (SD={std_acc:.2f}%)")
print(f"Macro F1-Score:    {m_f1m:.2f}% [95% CI: {low_f1m:.2f}% - {high_f1m:.2f}%] (SD={std_f1m:.2f}%)")
print(f"Weighted F1-Score: {m_f1w:.2f}% [95% CI: {low_f1w:.2f}% - {high_f1w:.2f}%] (SD={std_f1w:.2f}%)")
print(f"Expected Risk/Tap: {m_risk:.4f} [95% CI: {low_risk:.4f} - {high_risk:.4f}] (SD={std_risk:.4f})")
print(f"Sitting Accuracy:  {m_sit:.2f}% [95% CI: {low_sit:.2f}% - {high_sit:.2f}%] (SD={std_sit:.2f}%)")
print(f"Walking Accuracy:  {m_walk:.2f}% [95% CI: {low_walk:.2f}% - {high_walk:.2f}%] (SD={std_walk:.2f}%)")

# Per-class metrics with 95% CI
class_names = ['Small (40dp)', 'Default (48dp)', 'Large (56dp)', 'Extra Large (64dp)']
per_class_results = []

for c in range(4):
    boot_prec = []
    boot_rec = []
    boot_f1 = []
    c_mask = (y_test == c).values
    for _ in range(B):
        idx = np.random.choice(N, size=N, replace=True)
        y_t_b = (y_test.iloc[idx] == c).astype(int)
        y_p_b = (y_pred[idx] == c).astype(int)
        boot_prec.append(precision_score(y_t_b, y_p_b, zero_division=0) * 100.0)
        boot_rec.append(recall_score(y_t_b, y_p_b, zero_division=0) * 100.0)
        boot_f1.append(f1_score(y_t_b, y_p_b, zero_division=0) * 100.0)
    
    mp, lp, hp, _ = get_ci(boot_prec)
    mr, lr, hr, _ = get_ci(boot_rec)
    mf, lf, hf, _ = get_ci(boot_f1)
    
    per_class_results.append({
        'Class ID': c,
        'Class Name': class_names[c],
        'Precision (%)': f"{mp:.2f}% [{lp:.2f}%, {hp:.2f}%]",
        'Recall (%)': f"{mr:.2f}% [{lr:.2f}%, {hr:.2f}%]",
        'F1-Score (%)': f"{mf:.2f}% [{lf:.2f}%, {hf:.2f}%]",
        'Mean_F1': mf,
        'Low_F1': lf,
        'High_F1': hf
    })

df_per_class_ci = pd.DataFrame(per_class_results)
df_per_class_ci.to_csv(os.path.join(outputs_dir, "04_confidence_intervals_per_class.csv"), index=False)
print("\nPer-class 95% CI:")
print(df_per_class_ci[['Class Name', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']])

# Save master metrics table
ci_master_summary = pd.DataFrame([
    {'Metric': 'Overall Test Accuracy (Bootstrap)', 'Mean': f"{m_acc:.2f}%", '95% CI Lower': f"{low_acc:.2f}%", '95% CI Upper': f"{high_acc:.2f}%", 'Margin': f"±{(high_acc-low_acc)/2:.2f}%"},
    {'Metric': 'Overall Test Accuracy (Wilson Score)', 'Mean': f"{p_hat*100:.2f}%", '95% CI Lower': f"{wilson_low:.2f}%", '95% CI Upper': f"{wilson_high:.2f}%", 'Margin': f"±{(wilson_high-wilson_low)/2:.2f}%"},
    {'Metric': 'Macro F1-Score', 'Mean': f"{m_f1m:.2f}%", '95% CI Lower': f"{low_f1m:.2f}%", '95% CI Upper': f"{high_f1m:.2f}%", 'Margin': f"±{(high_f1m-low_f1m)/2:.2f}%"},
    {'Metric': 'Weighted F1-Score', 'Mean': f"{m_f1w:.2f}%", '95% CI Lower': f"{low_f1w:.2f}%", '95% CI Upper': f"{high_f1w:.2f}%", 'Margin': f"±{(high_f1w-low_f1w)/2:.2f}%"},
    {'Metric': 'Sitting Context Accuracy', 'Mean': f"{m_sit:.2f}%", '95% CI Lower': f"{low_sit:.2f}%", '95% CI Upper': f"{high_sit:.2f}%", 'Margin': f"±{(high_sit-low_sit)/2:.2f}%"},
    {'Metric': 'Walking Context Accuracy', 'Mean': f"{m_walk:.2f}%", '95% CI Lower': f"{low_walk:.2f}%", '95% CI Upper': f"{high_walk:.2f}%", 'Margin': f"±{(high_walk-low_walk)/2:.2f}%"},
    {'Metric': 'Interaction Risk (Cost per Tap)', 'Mean': f"{m_risk:.4f}", '95% CI Lower': f"{low_risk:.4f}", '95% CI Upper': f"{high_risk:.4f}", 'Margin': f"±{(high_risk-low_risk)/2:.4f}"}
])
ci_master_summary.to_csv(os.path.join(outputs_dir, "04_confidence_intervals_summary.csv"), index=False)

# Multi-model 10-Seed Confidence Intervals
# Re-extract from the multi-seed evaluation
SEEDS = [0, 7, 21, 42, 99, 123, 256, 512, 999, 2024]
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

train_path = os.path.join(base_dir, "processed", "final_ready_dataset.csv")
df_train = pd.read_csv(train_path)
X_train = df_train[FEATURE_COLS].copy()
y_train = df_train[TARGET_COL].copy().astype(int)

# Fast benchmark of top models across seeds to get exact Student's t 95% CIs
models_dict = {
    'HistGradientBoosting': [],
    'Hybrid Ensemble': [],
    'LightGBM': [],
    'CatBoost': [],
    'XGBoost': [],
    'Random Forest': []
}

print("\nComputing multi-seed 95% CIs across 10 independent random seeds...")
for seed in SEEDS:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    tr_idx, va_idx = next(sss.split(X_train, y_train))
    scl = RobustScaler()
    X_tr_s = scl.fit_transform(X_train.iloc[tr_idx])
    X_ext_s = scl.transform(X_test)
    y_tr_s = y_train.iloc[tr_idx]
    
    # 1. HistGB
    hgb = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.08, random_state=seed)
    hgb.fit(X_tr_s, y_tr_s)
    models_dict['HistGradientBoosting'].append(accuracy_score(y_test, hgb.predict(X_ext_s)) * 100.0)
    
    # 2. LGBM
    lgb = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, subsample=0.9, subsample_freq=1, colsample_bytree=0.9, random_state=seed, verbose=-1)
    lgb.fit(X_tr_s, y_tr_s)
    lgb_preds = lgb.predict(X_ext_s)
    models_dict['LightGBM'].append(accuracy_score(y_test, lgb_preds) * 100.0)
    
    # 3. CatBoost
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.08, random_seed=seed, verbose=0)
    cat.fit(X_tr_s, y_tr_s)
    cat_preds = cat.predict(X_ext_s)
    models_dict['CatBoost'].append(accuracy_score(y_test, cat_preds) * 100.0)
    
    # 4. XGBoost
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, eval_metric='mlogloss', random_state=seed)
    xgb.fit(X_tr_s, y_tr_s)
    xgb_preds = xgb.predict(X_ext_s)
    models_dict['XGBoost'].append(accuracy_score(y_test, xgb_preds) * 100.0)
    
    # Hybrid soft voting
    p_xgb = xgb.predict_proba(X_ext_s)
    p_lgb = lgb.predict_proba(X_ext_s)
    p_cat = cat.predict_proba(X_ext_s)
    p_ens = (p_xgb + p_lgb + p_cat) / 3.0
    models_dict['Hybrid Ensemble'].append(accuracy_score(y_test, np.argmax(p_ens, axis=1)) * 100.0)
    
    # RF
    rf = RandomForestClassifier(n_estimators=180, max_depth=10, random_state=seed)
    rf.fit(X_tr_s, y_tr_s)
    models_dict['Random Forest'].append(accuracy_score(y_test, rf.predict(X_ext_s)) * 100.0)

t_crit = stats.t.ppf(0.975, df=len(SEEDS)-1) # 2.262
seed_ci_results = []
for m_name, accs in models_dict.items():
    mean_val = np.mean(accs)
    std_val = np.std(accs, ddof=1)
    se_val = std_val / np.sqrt(len(SEEDS))
    ci_low = mean_val - t_crit * se_val
    ci_high = mean_val + t_crit * se_val
    seed_ci_results.append({
        'Model': m_name,
        'Mean Accuracy (%)': mean_val,
        'Std Dev (%)': std_val,
        'Std Error (%)': se_val,
        '95% CI Lower (%)': ci_low,
        '95% CI Upper (%)': ci_high,
        'Formatted 95% CI': f"{mean_val:.2f}% [{ci_low:.2f}%, {ci_high:.2f}%]"
    })

df_seed_ci = pd.DataFrame(seed_ci_results)
df_seed_ci.to_csv(os.path.join(outputs_dir, "04_multiseed_10_seeds_ci_comparison.csv"), index=False)
print("\nMulti-Seed 95% Confidence Intervals across 10 Seeds:")
print(df_seed_ci[['Model', 'Formatted 95% CI', 'Std Dev (%)']])

# Plot Publication Figure
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Subplot 1: Bootstrap Distribution for HistGradientBoosting
ax1 = axes[0]
sns.histplot(boot_acc, kde=True, color='#1f77b4', bins=30, ax=ax1, stat='density', alpha=0.6)
ax1.axvline(m_acc, color='#004d80', linestyle='-', linewidth=2.5, label=f'Mean = {m_acc:.2f}%')
ax1.axvline(low_acc, color='#d62728', linestyle='--', linewidth=2, label=f'2.5% CI = {low_acc:.2f}%')
ax1.axvline(high_acc, color='#d62728', linestyle='--', linewidth=2, label=f'97.5% CI = {high_acc:.2f}%')
ax1.set_title('(A) Non-Parametric Bootstrap Test Accuracy\n(B = 2,000 resamples, N = 3,580 unseen taps)', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
ax1.legend(loc='upper right', frameon=True, fontsize=10)

# Subplot 2: Multi-Model 95% CI Error Bar Comparison (10 Seeds)
ax2 = axes[1]
y_positions = np.arange(len(df_seed_ci))
models_ordered = df_seed_ci['Model'].tolist()
means = df_seed_ci['Mean Accuracy (%)'].values
lows = df_seed_ci['95% CI Lower (%)'].values
highs = df_seed_ci['95% CI Upper (%)'].values
errors = [means - lows, highs - means]

colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#d62728']
for idx, (m_val, l_val, h_val, col) in enumerate(zip(means, lows, highs, colors)):
    ax2.errorbar(m_val, idx, xerr=[[m_val - l_val], [h_val - m_val]], fmt='o', color=col, ecolor=col, elinewidth=2.5, capsize=6, capthick=2, markersize=8)
    ax2.text(m_val, idx + 0.22, f"{m_val:.2f}% [{l_val:.2f}%, {h_val:.2f}%]", ha='center', va='bottom', fontsize=9, fontweight='bold', color=col)

ax2.set_yticks(y_positions)
ax2.set_yticklabels(models_ordered, fontsize=11, fontweight='bold')
ax2.set_xlabel('Test Accuracy with 95% Confidence Interval (%)', fontsize=11, fontweight='bold')
ax2.set_title('(B) Multi-Seed Stability (10 Random Seeds)\nStudent\'s t-distribution 95% CI (df=9)', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlim(85, 93)

# Subplot 3: Per-Class F1-Scores with 95% CI
ax3 = axes[2]
c_means = df_per_class_ci['Mean_F1'].values
c_lows = df_per_class_ci['Low_F1'].values
c_highs = df_per_class_ci['High_F1'].values
c_errs = [c_means - c_lows, c_highs - c_means]
bar_colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

bars = ax3.bar(range(4), c_means, yerr=c_errs, capsize=6, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=1.2)
for idx, (bar, m_val, l_val, h_val) in enumerate(zip(bars, c_means, c_lows, c_highs)):
    ax3.text(bar.get_x() + bar.get_width() / 2, m_val / 2, f"{m_val:.1f}%\n[{l_val:.1f}%, {h_val:.1f}%]", ha='center', va='center', fontsize=9.5, fontweight='bold', color='white')

ax3.set_xticks(range(4))
ax3.set_xticklabels(['Small\n(40dp)', 'Default\n(48dp)', 'Large\n(56dp)', 'Extra Large\n(64dp)'], fontsize=10, fontweight='bold')
ax3.set_ylabel('F1-Score (%)', fontsize=11, fontweight='bold')
ax3.set_title('(C) Per-Class Adaptation F1-Scores\n(2,000 Bootstrap 95% CI)', fontsize=12, fontweight='bold', pad=10)
ax3.set_ylim(0, 105)

plt.tight_layout()
fig_path = os.path.join(figures_dir, "fig_confidence_intervals_evaluation.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved CI figure to: {fig_path}")
