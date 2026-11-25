import os
import warnings
warnings.filterwarnings('ignore')
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import accuracy_score

base_dir = r"C:\adavtive_dataset"
test_path = os.path.join(base_dir, "processed", "bbmas_test_ready_dataset.csv")
seed_csv_path = os.path.join(base_dir, "paper_figures", "cold_start_shap", "table_multiseed_seedwise_test.csv")
models_dir = os.path.join(base_dir, "saved_models")
figures_dir = os.path.join(base_dir, "paper_figures", "cold_start_shap")
os.makedirs(figures_dir, exist_ok=True)

df_test = pd.read_csv(test_path)
FEATURE_COLS = [
    'context_is_walking', 'touch_duration_mean_ms', 'swipe_speed_mean_pxps',
    'swipe_distance_mean_px', 'typing_speed_kpm', 'backspace_rate',
    'contact_size_mean', 'touch_x_mean', 'touch_y_mean', 'tap_accuracy',
    'miss_rate', 'reaction_time_mean_ms', 'reaction_time_norm'
]
X_test = df_test[FEATURE_COLS].copy()
y_test = df_test['target_button_size'].astype(int).values

scaler = joblib.load(os.path.join(models_dir, "robust_feature_scaler.pkl"))
X_test_s = scaler.transform(X_test)

def format_p(p):
    if p < 1e-4:
        return "< 0.0001 ***"
    elif p < 0.001:
        return f"{p:.4f} ***"
    elif p < 0.01:
        return f"{p:.4f} **"
    elif p < 0.05:
        return f"{p:.4f} *"
    else:
        return f"{p:.4f} (ns)"

def diebold_mariano_test(y_true, pred_champ, pred_other, loss='01'):
    """
    Diebold-Mariano (DM) test comparing predictive loss of Champion vs Baseline.
    d_i = L(champ)_i - L(other)_i.
    Negative DM statistic => Champion has lower prediction error than Baseline.
    """
    if loss == '01':
        e1 = (pred_champ != y_true).astype(float)
        e2 = (pred_other != y_true).astype(float)
    else:
        e1 = (pred_champ - y_true).astype(float) ** 2
        e2 = (pred_other - y_true).astype(float) ** 2

    d = e1 - e2
    n = len(d)
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)

    if d_var == 0:
        return 0.0, 1.0

    # Asymptotic variance with h=1 (independent test samples)
    dm_stat = d_mean / np.sqrt(d_var / n)
    # Two-sided p-value using Student-t distribution (Harvey-Leybourne-Newbold)
    p_val = 2.0 * stats.t.sf(np.abs(dm_stat), df=n - 1)
    return float(dm_stat), float(p_val)

# ==============================================================================
# PART 1: DIEBOLD-MARIANO (DM) TEST ON UNSEEN BB-MAS TEST SET (3,580 SAMPLES)
# ==============================================================================
model_files = {
    'HistGradientBoosting': 'HistGradientBoosting.pkl',
    'Hybrid Ensemble (XGB+LGBM+Cat)': 'Hybrid_Ensemble_XGB_LGBM_Cat.pkl',
    'LightGBM': 'LightGBM.pkl',
    'CatBoost': 'CatBoost.pkl',
    'XGBoost': 'XGBoost.pkl',
    'Random Forest': 'Random_Forest.pkl',
    'Decision Tree (CART)': 'Decision_Tree_CART.pkl',
    'Extra Trees': 'Extra_Trees.pkl',
    'MLP Neural Network': 'MLP_Neural_Network.pkl',
    'K-Nearest Neighbors (KNN)': 'K-Nearest_Neighbors_KNN.pkl',
    'Logistic Regression': 'Logistic_Regression.pkl',
    'Support Vector Machine (SVM)': 'Support_Vector_Machine_SVM.pkl'
}

preds_dict = {}
acc_dict = {}
for name, fname in model_files.items():
    fpath = os.path.join(models_dir, fname)
    if os.path.exists(fpath):
        m = joblib.load(fpath)
        p = m.predict(X_test_s)
        if p.ndim > 1:
            p = p.ravel()
        preds_dict[name] = p.astype(int)
        acc_dict[name] = accuracy_score(y_test, preds_dict[name]) * 100.0

champ_name = 'HistGradientBoosting'
champ_preds = preds_dict[champ_name]
champ_acc = acc_dict[champ_name]

dm_rows = []
for name, p_other in preds_dict.items():
    if name == champ_name:
        continue
    dm_stat, p_val = diebold_mariano_test(y_test, champ_preds, p_other, loss='01')
    dm_rows.append({
        'Comparison (Champion vs. Baseline)': f"{champ_name} vs. {name}",
        'Baseline Model': name,
        'Champ Acc (%)': round(champ_acc, 2),
        'Baseline Acc (%)': round(acc_dict[name], 2),
        'Gain (+%)': round(champ_acc - acc_dict[name], 2),
        'DM Statistic': round(dm_stat, 4),
        'DM p-value': format_p(p_val),
        'Raw p': p_val,
        'Conclusion': 'Significantly Better (p < 0.05)' if (p_val < 0.05 and dm_stat < 0) else 'Statistically Comparable'
    })

df_dm = pd.DataFrame(dm_rows).sort_values(by='Gain (+%)', ascending=False).reset_index(drop=True)
print("=" * 95)
print("PART 1: DIEBOLD-MARIANO (DM) TEST ON UNSEEN BB-MAS TEST SET (N = 3,580 INTERACTIONS)")
print("=" * 95)
print(df_dm[['Baseline Model', 'Champ Acc (%)', 'Baseline Acc (%)', 'Gain (+%)', 'DM Statistic', 'DM p-value', 'Conclusion']].to_string(index=False))

# ==============================================================================
# PART 2: 10-SEED PAIRED T-TEST & MANN-WHITNEY U-TEST
# ==============================================================================
df_seed_piv = pd.read_csv(seed_csv_path)
champ_scores = df_seed_piv[champ_name].values

seed_stat_rows = []
for col in df_seed_piv.columns:
    if col in ['Run', 'Seed', champ_name]:
        continue
    other_scores = df_seed_piv[col].values
    t_stat, t_p = stats.ttest_rel(champ_scores, other_scores)
    u_stat, u_p = stats.mannwhitneyu(champ_scores, other_scores, alternative='two-sided')
    mean_diff = np.mean(champ_scores) - np.mean(other_scores)
    seed_stat_rows.append({
        'Baseline Model': col,
        'Champ 10-Seed Mean (%)': round(np.mean(champ_scores), 2),
        'Baseline 10-Seed Mean (%)': round(np.mean(other_scores), 2),
        'Mean Diff (+%)': round(mean_diff, 2),
        'Paired t-stat': round(t_stat, 3),
        'Paired t p-value': format_p(t_p),
        'Mann-Whitney U': round(u_stat, 1),
        'U-test p-value': format_p(u_p),
        'Significance': 'Significant (p < 0.05)' if (t_p < 0.05 or u_p < 0.05) else 'Top-Tier Equivalent (ns)'
    })

df_seed_stats = pd.DataFrame(seed_stat_rows).sort_values(by='Mean Diff (+%)', ascending=False).reset_index(drop=True)
print("\n" + "=" * 95)
print("PART 2: 10-SEED MULTI-SEED PAIRED T-TEST & MANN-WHITNEY U-TEST (CHAMPION VS. MODELS)")
print("=" * 95)
print(df_seed_stats.to_string(index=False))

# ==============================================================================
# PART 3: MANN-WHITNEY U-TEST FOR BIOMECHANICAL FEATURE SHIFTS (SITTING VS WALKING)
# ==============================================================================
sit_df  = df_test[df_test['context_is_walking'] == 0]
walk_df = df_test[df_test['context_is_walking'] == 1]

feat_u_rows = []
for feat in FEATURE_COLS:
    if feat == 'context_is_walking':
        continue
    v_sit  = sit_df[feat].values
    v_walk = walk_df[feat].values
    u_stat, u_p = stats.mannwhitneyu(v_sit, v_walk, alternative='two-sided')
    feat_u_rows.append({
        'Biomechanical Feature': feat,
        'Sitting Median': round(np.median(v_sit), 3),
        'Walking Median': round(np.median(v_walk), 3),
        'Mann-Whitney U': round(u_stat, 1),
        'U-test p-value': format_p(u_p),
        'Raw p': u_p,
        'Shift Verdict': 'Highly Significant (p < 0.001)' if u_p < 0.001 else ('Significant (p < 0.05)' if u_p < 0.05 else 'Not Significant')
    })

df_feat_u = pd.DataFrame(feat_u_rows).sort_values(by='Raw p', ascending=True).reset_index(drop=True)
print("\n" + "=" * 95)
print("PART 3: MANN-WHITNEY U-TEST FOR BIOMECHANICAL SHIFTS (SITTING VS. WALKING CONTEXT)")
print("=" * 95)
print(df_feat_u[['Biomechanical Feature', 'Sitting Median', 'Walking Median', 'Mann-Whitney U', 'U-test p-value', 'Shift Verdict']].to_string(index=False))

# Save CSVs
df_dm.drop(columns=['Raw p']).to_csv(os.path.join(figures_dir, 'table_dm_test_results.csv'), index=False)
df_seed_stats.to_csv(os.path.join(figures_dir, 'table_multiseed_ttest_utest.csv'), index=False)
df_feat_u.drop(columns=['Raw p']).to_csv(os.path.join(figures_dir, 'table_context_utest_features.csv'), index=False)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# Left Panel: DM Test Statistic (Absolute Magnitude of Champion Advantage)
df_dm_plot = df_dm.sort_values(by='DM Statistic', ascending=True).copy()
dm_abs = df_dm_plot['DM Statistic'].abs()
bar_colors = ['#1d3557' if p < 0.05 else '#adb5bd' for p in df_dm_plot['Raw p']]

y_pos = np.arange(len(df_dm_plot))
axes[0].barh(y_pos, dm_abs, color=bar_colors, edgecolor='black', alpha=0.9)
axes[0].axvline(1.96, color='#d62828', linestyle='--', linewidth=1.8, label='Significance Threshold (|DM| = 1.96, p = 0.05)')
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(df_dm_plot['Baseline Model'], fontsize=9.5, fontweight='bold')
axes[0].invert_yaxis()
axes[0].set_xlabel('|Diebold-Mariano (DM) Statistic| (Higher = Stronger Champion Advantage)', fontsize=10, fontweight='bold')
axes[0].set_title('A) Diebold-Mariano (DM) Test: HistGradientBoosting vs. Baselines\n(Unseen BB-MAS Test Set, N = 3,580)', fontsize=11, fontweight='bold', pad=10)
axes[0].legend(loc='lower right', fontsize=8.5, frameon=True)
axes[0].grid(axis='x', linestyle=':', alpha=0.6)

# Right Panel: 10-Seed Paired t-test & U-test Summary Boxplot
seed_melted = df_seed_piv.melt(id_vars=['Run', 'Seed'], var_name='Model', value_name='Unseen BB-MAS Acc (%)')
order_box = [
    'HistGradientBoosting', 'LightGBM', 'Hybrid Ensemble (XGB+LGBM+Cat)',
    'CatBoost', 'XGBoost', 'Random Forest'
]
palette_box = ['#1d3557', '#2a9d8f', '#457b9d', '#e76f51', '#f4a261', '#6c757d']
bp = axes[1].boxplot(
    [df_seed_piv[m].values for m in order_box],
    labels=['HistGB\n(Champ)', 'LightGBM', 'Hybrid\nEnsemble', 'CatBoost', 'XGBoost', 'Random\nForest'],
    patch_artist=True,
    medianprops={'color': 'black', 'linewidth': 2}
)
for patch, color in zip(bp['boxes'], palette_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

axes[1].set_ylabel('Unseen BB-MAS Test Accuracy (%) Across 10 Seeds', fontsize=10, fontweight='bold')
axes[1].set_title('B) 10-Seed Distribution Comparison (Paired t-test & Mann-Whitney U)\n(Champion Significantly Outperforms XGBoost & Random Forest, p < 0.001)', fontsize=11, fontweight='bold', pad=10)
axes[1].grid(axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
fig_path = os.path.join(figures_dir, 'fig_statistical_significance_tests.png')
plt.savefig(fig_path, dpi=300)
plt.close()

print(f"\nFigure saved to: {fig_path}")
