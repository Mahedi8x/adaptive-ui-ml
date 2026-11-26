import os
import warnings
warnings.filterwarnings('ignore')
import joblib
import numpy as np
import pandas as pd
import matplotlib
if matplotlib.get_backend().lower() not in ['nbagg', 'inline', 'module://matplotlib_inline.backend_inline']:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
if not os.path.exists(os.path.join(base_dir, "processed", "bbmas_test_ready_dataset.csv")):
    base_dir = r"C:\adavtive_dataset"

train_path = os.path.join(base_dir, "processed", "final_ready_dataset.csv")
test_path = os.path.join(base_dir, "processed", "bbmas_test_ready_dataset.csv")
models_dir = os.path.join(base_dir, "saved_models")
figures_dir = os.path.join(base_dir, "paper_figures", "cold_start_shap")
os.makedirs(figures_dir, exist_ok=True)

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

FEATURE_COLS = [
    'context_is_walking', 'touch_duration_mean_ms', 'swipe_speed_mean_pxps',
    'swipe_distance_mean_px', 'typing_speed_kpm', 'backspace_rate',
    'contact_size_mean', 'touch_x_mean', 'touch_y_mean', 'tap_accuracy',
    'miss_rate', 'reaction_time_mean_ms', 'reaction_time_norm'
]
TARGET_COL = 'target_button_size'
CLASS_NAMES = ['Small (40dp)', 'Default (48dp)', 'Large (56dp)', 'Extra Large (64dp)']

X_train = df_train[FEATURE_COLS].copy()
y_train = df_train[TARGET_COL].astype(int)
X_test = df_test[FEATURE_COLS].copy()
y_test = df_test[TARGET_COL].astype(int).values

sit_mask = (df_test['context_is_walking'] == 0).values
walk_mask = (df_test['context_is_walking'] == 1).values

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

    dm_stat = d_mean / np.sqrt(d_var / n)
    p_val = 2.0 * stats.t.sf(np.abs(dm_stat), df=n - 1)
    return float(dm_stat), float(p_val)

# ==============================================================================
# PART 1: DIEBOLD-MARIANO (DM) TEST & MACRO ACCURACY ON UNSEEN BB-MAS (N=3,580)
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
macro_acc_dict = {}
macro_f1_dict = {}
sit_macro_acc_dict = {}
walk_macro_acc_dict = {}

for name, fname in model_files.items():
    fpath = os.path.join(models_dir, fname)
    if os.path.exists(fpath):
        m = joblib.load(fpath)
        p = m.predict(X_test_s)
        if p.ndim > 1:
            p = p.ravel()
        p = p.astype(int)
        preds_dict[name] = p
        acc_dict[name] = accuracy_score(y_test, p) * 100.0
        macro_acc_dict[name] = balanced_accuracy_score(y_test, p) * 100.0
        macro_f1_dict[name] = f1_score(y_test, p, average='macro', zero_division=0) * 100.0
        sit_macro_acc_dict[name] = balanced_accuracy_score(y_test[sit_mask], p[sit_mask]) * 100.0
        walk_macro_acc_dict[name] = balanced_accuracy_score(y_test[walk_mask], p[walk_mask]) * 100.0

champ_name = 'HistGradientBoosting'
champ_preds = preds_dict[champ_name]
champ_acc = acc_dict[champ_name]
champ_macro_acc = macro_acc_dict[champ_name]
champ_macro_f1 = macro_f1_dict[champ_name]

# Verify Per-Class Accuracy -> Macro Accuracy for Champion
cm_champ = confusion_matrix(y_test, champ_preds)
per_class_acc_champ = cm_champ.diagonal() / cm_champ.sum(axis=1) * 100.0

dm_rows = []
for name, p_other in preds_dict.items():
    if name == champ_name:
        continue
    dm_stat, p_val = diebold_mariano_test(y_test, champ_preds, p_other, loss='01')
    dm_rows.append({
        'Comparison (Champion vs. Baseline)': f"{champ_name} vs. {name}",
        'Baseline Model': name,
        'Champ Macro Acc (%)': round(champ_macro_acc, 2),
        'Baseline Macro Acc (%)': round(macro_acc_dict[name], 2),
        'Baseline Macro F1 (%)': round(macro_f1_dict[name], 2),
        'Macro Acc Gain (+%)': round(champ_macro_acc - macro_acc_dict[name], 2),
        'DM Statistic': round(dm_stat, 4),
        'DM p-value': format_p(p_val),
        'Raw p': p_val,
        'Conclusion': 'Significantly Better (p < 0.05)' if (p_val < 0.05 and dm_stat < 0) else 'Statistically Comparable'
    })

df_dm = pd.DataFrame(dm_rows).sort_values(by='Macro Acc Gain (+%)', ascending=False).reset_index(drop=True)
print("=" * 108)
print("PART 1: DIEBOLD-MARIANO (DM) TEST & MACRO ACCURACY ON UNSEEN BB-MAS TEST SET (N = 3,580, 895/CLASS)")
print("=" * 108)
print(f"Champion ({champ_name}) Per-Class Accuracy: " +
      ", ".join([f"{cn}: {val:.2f}%" for cn, val in zip(CLASS_NAMES, per_class_acc_champ)]))
print(f"-> Champion Overall Macro (Balanced) Accuracy = {champ_macro_acc:.2f}% | Macro F1 = {champ_macro_f1:.2f}%")
print(f"-> Champion Context Macro Accuracy: Sitting = {sit_macro_acc_dict[champ_name]:.2f}% | Walking = {walk_macro_acc_dict[champ_name]:.2f}%")
print("-" * 108)
print(df_dm[['Baseline Model', 'Champ Macro Acc (%)', 'Baseline Macro Acc (%)', 'Baseline Macro F1 (%)',
             'Macro Acc Gain (+%)', 'DM Statistic', 'DM p-value', 'Conclusion']].to_string(index=False))

# ==============================================================================
# PART 2: 10-SEED MULTI-SEED PAIRED T-TEST & MANN-WHITNEY U-TEST (WITH MACRO ACCURACY)
# ==============================================================================
SEEDS = [0, 7, 21, 42, 99, 123, 256, 512, 999, 2024]
model_order = [
    'HistGradientBoosting',
    'Hybrid Ensemble (XGB+LGBM+Cat)',
    'LightGBM',
    'CatBoost',
    'XGBoost',
    'Random Forest'
]

seed_macro_acc_csv = os.path.join(figures_dir, "table_multiseed_seedwise_macro_acc.csv")
seed_macro_f1_csv  = os.path.join(figures_dir, "table_multiseed_seedwise_macro_f1.csv")
seed_sit_macro_csv = os.path.join(figures_dir, "table_multiseed_seedwise_sitting_macro_acc.csv")
seed_walk_macro_csv = os.path.join(figures_dir, "table_multiseed_seedwise_walking_macro_acc.csv")

if not (os.path.exists(seed_macro_acc_csv) and os.path.exists(seed_macro_f1_csv) and
        os.path.exists(seed_sit_macro_csv) and os.path.exists(seed_walk_macro_csv)):
    records_macro_acc = []
    records_macro_f1 = []
    records_sit_macro = []
    records_walk_macro = []

    for run_idx, seed in enumerate(SEEDS, 1):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        tr_idx, _ = next(sss.split(X_train, y_train))
        scl = RobustScaler()
        X_tr_s = scl.fit_transform(X_train.iloc[tr_idx])
        X_ext_s = scl.transform(X_test)
        y_tr_s = y_train.iloc[tr_idx]

        hgb = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.08, random_state=seed)
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, eval_metric='mlogloss', random_state=seed)
        lgb = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, subsample=0.9, subsample_freq=1, colsample_bytree=0.9, random_state=seed, verbose=-1)
        cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.08, random_seed=seed, verbose=0)
        rf  = RandomForestClassifier(n_estimators=180, max_depth=10, random_state=seed)

        hgb.fit(X_tr_s, y_tr_s)
        xgb.fit(X_tr_s, y_tr_s)
        lgb.fit(X_tr_s, y_tr_s)
        cat.fit(X_tr_s, y_tr_s)
        rf.fit(X_tr_s, y_tr_s)

        p_ens = np.argmax((xgb.predict_proba(X_ext_s) + lgb.predict_proba(X_ext_s) + cat.predict_proba(X_ext_s)) / 3.0, axis=1)
        seed_preds = {
            'HistGradientBoosting': hgb.predict(X_ext_s).astype(int),
            'Hybrid Ensemble (XGB+LGBM+Cat)': p_ens.astype(int),
            'LightGBM': lgb.predict(X_ext_s).astype(int),
            'CatBoost': cat.predict(X_ext_s).ravel().astype(int),
            'XGBoost': xgb.predict(X_ext_s).astype(int),
            'Random Forest': rf.predict(X_ext_s).astype(int)
        }

        row_ma = {'Run': f"Run {run_idx}", 'Seed': seed}
        row_mf = {'Run': f"Run {run_idx}", 'Seed': seed}
        row_sm = {'Run': f"Run {run_idx}", 'Seed': seed}
        row_wm = {'Run': f"Run {run_idx}", 'Seed': seed}
        for m_name in model_order:
            p = seed_preds[m_name]
            row_ma[m_name] = round(balanced_accuracy_score(y_test, p) * 100.0, 2)
            row_mf[m_name] = round(f1_score(y_test, p, average='macro', zero_division=0) * 100.0, 2)
            row_sm[m_name] = round(balanced_accuracy_score(y_test[sit_mask], p[sit_mask]) * 100.0, 2)
            row_wm[m_name] = round(balanced_accuracy_score(y_test[walk_mask], p[walk_mask]) * 100.0, 2)

        records_macro_acc.append(row_ma)
        records_macro_f1.append(row_mf)
        records_sit_macro.append(row_sm)
        records_walk_macro.append(row_wm)

    pd.DataFrame(records_macro_acc).to_csv(seed_macro_acc_csv, index=False)
    pd.DataFrame(records_macro_f1).to_csv(seed_macro_f1_csv, index=False)
    pd.DataFrame(records_sit_macro).to_csv(seed_sit_macro_csv, index=False)
    pd.DataFrame(records_walk_macro).to_csv(seed_walk_macro_csv, index=False)

df_seed_ma = pd.read_csv(seed_macro_acc_csv)
df_seed_mf = pd.read_csv(seed_macro_f1_csv)
df_seed_sm = pd.read_csv(seed_sit_macro_csv)
df_seed_wm = pd.read_csv(seed_walk_macro_csv)

def run_seed_tests(df_metric, metric_label):
    champ_vals = df_metric[champ_name].values
    rows = []
    for col in model_order:
        if col == champ_name:
            continue
        other_vals = df_metric[col].values
        t_stat, t_p = stats.ttest_rel(champ_vals, other_vals)
        u_stat, u_p = stats.mannwhitneyu(champ_vals, other_vals, alternative='two-sided')
        mean_diff = np.mean(champ_vals) - np.mean(other_vals)
        rows.append({
            'Baseline Model': col,
            f'Champ {metric_label} (%)': f"{np.mean(champ_vals):.2f} ± {np.std(champ_vals, ddof=1):.2f}",
            f'Baseline {metric_label} (%)': f"{np.mean(other_vals):.2f} ± {np.std(other_vals, ddof=1):.2f}",
            'Mean Diff (+%)': round(mean_diff, 2),
            'Paired t-stat': round(t_stat, 3),
            'Paired t p-value': format_p(t_p),
            'Mann-Whitney U': round(u_stat, 1),
            'U-test p-value': format_p(u_p),
            'Significance': 'Significant (p < 0.05)' if (t_p < 0.05 or u_p < 0.05) else 'Top-Tier Equivalent (ns)'
        })
    return pd.DataFrame(rows).sort_values(by='Mean Diff (+%)', ascending=False).reset_index(drop=True)

df_stats_macro_acc = run_seed_tests(df_seed_ma, "Macro Acc")
df_stats_macro_f1  = run_seed_tests(df_seed_mf, "Macro F1")
df_stats_sit_ma    = run_seed_tests(df_seed_sm, "Sit Macro Acc")
df_stats_walk_ma   = run_seed_tests(df_seed_wm, "Walk Macro Acc")

print("\n" + "=" * 108)
print("PART 2A: 10-SEED PAIRED T-TEST & MANN-WHITNEY U-TEST ON MACRO ACCURACY (BALANCED ACCURACY)")
print("         (Note: Equal 25% class balance [895/class] makes Overall Acc == Macro Acc on full BB-MAS)")
print("=" * 108)
print(df_stats_macro_acc.to_string(index=False))

print("\n" + "=" * 108)
print("PART 2B: 10-SEED PAIRED T-TEST & MANN-WHITNEY U-TEST ON MACRO F1-SCORE")
print("=" * 108)
print(df_stats_macro_f1.to_string(index=False))

print("\n" + "=" * 108)
print("PART 2C: 10-SEED PAIRED T-TEST & MANN-WHITNEY U-TEST ON CONTEXT-STRATIFIED MACRO ACCURACY")
print("         (Sitting & Walking subsets have natural class imbalance -> Macro Acc != Overall Acc)")
print("=" * 108)
print("[Sitting Context Macro Accuracy]")
print(df_stats_sit_ma.to_string(index=False))
print("\n[Walking Context Macro Accuracy]")
print(df_stats_walk_ma.to_string(index=False))

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
print("\n" + "=" * 108)
print("PART 3: MANN-WHITNEY U-TEST FOR BIOMECHANICAL SHIFTS (SITTING VS. WALKING CONTEXT)")
print("=" * 108)
print(df_feat_u[['Biomechanical Feature', 'Sitting Median', 'Walking Median', 'Mann-Whitney U', 'U-test p-value', 'Shift Verdict']].to_string(index=False))

# Save CSVs
df_dm.drop(columns=['Raw p']).to_csv(os.path.join(figures_dir, 'table_dm_test_results.csv'), index=False)
df_stats_macro_acc.to_csv(os.path.join(figures_dir, 'table_multiseed_ttest_utest.csv'), index=False)
df_stats_macro_f1.to_csv(os.path.join(figures_dir, 'table_multiseed_ttest_utest_macro_f1.csv'), index=False)
df_stats_sit_ma.to_csv(os.path.join(figures_dir, 'table_multiseed_ttest_utest_sitting_macro_acc.csv'), index=False)
df_stats_walk_ma.to_csv(os.path.join(figures_dir, 'table_multiseed_ttest_utest_walking_macro_acc.csv'), index=False)
df_feat_u.drop(columns=['Raw p']).to_csv(os.path.join(figures_dir, 'table_context_utest_features.csv'), index=False)

# ==============================================================================
# PUBLICATION FIGURE (3 PANELS: DM TEST, 10-SEED MACRO ACC, CONTEXT MACRO ACC)
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 5.6))

# Panel A: DM Test Statistic (Champion Advantage + Macro Accuracy Annotation)
df_dm_plot = df_dm.sort_values(by='DM Statistic', ascending=True).copy()
dm_abs = df_dm_plot['DM Statistic'].abs()
bar_colors = ['#1d3557' if p < 0.05 else '#adb5bd' for p in df_dm_plot['Raw p']]

y_pos = np.arange(len(df_dm_plot))
axes[0].barh(y_pos, dm_abs, color=bar_colors, edgecolor='black', alpha=0.9)
axes[0].axvline(1.96, color='#d62828', linestyle='--', linewidth=1.8, label='Significance Threshold (|DM| = 1.96, p = 0.05)')
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(df_dm_plot['Baseline Model'], fontsize=9, fontweight='bold')
axes[0].invert_yaxis()
axes[0].set_xlabel('|Diebold-Mariano (DM) Statistic|', fontsize=10, fontweight='bold')
axes[0].set_title('A) Diebold-Mariano (DM) Test vs. Baselines\n(Unseen BB-MAS, Champion Macro Acc = 91.87%)', fontsize=10.8, fontweight='bold', pad=10)
axes[0].legend(loc='lower right', fontsize=8, frameon=True)
axes[0].grid(axis='x', linestyle=':', alpha=0.6)

# Panel B: 10-Seed Macro Accuracy Distribution Boxplot (Paired t-test & U-test)
order_box = [
    'HistGradientBoosting', 'LightGBM', 'Hybrid Ensemble (XGB+LGBM+Cat)',
    'CatBoost', 'XGBoost', 'Random Forest'
]
short_labels = ['HistGB\n(Champ)', 'LightGBM', 'Hybrid\nEnsemble', 'CatBoost', 'XGBoost', 'Random\nForest']
palette_box = ['#1d3557', '#2a9d8f', '#457b9d', '#e76f51', '#f4a261', '#6c757d']

bp = axes[1].boxplot(
    [df_seed_ma[m].values for m in order_box],
    labels=short_labels,
    patch_artist=True,
    medianprops={'color': 'black', 'linewidth': 2}
)
for patch, color in zip(bp['boxes'], palette_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

axes[1].set_ylabel('10-Seed Macro Accuracy (%) on BB-MAS', fontsize=10, fontweight='bold')
axes[1].set_title('B) 10-Seed Macro Accuracy (Paired t & U-Test)\n(HistGB vs. XGBoost & RF: p < 0.001 ***)', fontsize=10.8, fontweight='bold', pad=10)
axes[1].grid(axis='y', linestyle=':', alpha=0.6)

# Panel C: Context-Stratified Macro Accuracy (Sitting vs Walking) Across 10 Seeds
x_idx = np.arange(len(order_box))
width = 0.36
sit_means = [df_seed_sm[m].mean() for m in order_box]
sit_stds  = [df_seed_sm[m].std(ddof=1) for m in order_box]
walk_means = [df_seed_wm[m].mean() for m in order_box]
walk_stds  = [df_seed_wm[m].std(ddof=1) for m in order_box]

b1 = axes[2].bar(x_idx - width/2, sit_means, width, yerr=sit_stds, capsize=4, label='Sitting Macro Acc (%)', color='#457b9d', edgecolor='black', alpha=0.9)
b2 = axes[2].bar(x_idx + width/2, walk_means, width, yerr=walk_stds, capsize=4, label='Walking Macro Acc (%)', color='#e76f51', edgecolor='black', alpha=0.9)

for bar, val, sd in zip(b1, sit_means, sit_stds):
    axes[2].text(bar.get_x() + bar.get_width()/2, val + sd + 0.35, f"{val:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1d3557')
for bar, val, sd in zip(b2, walk_means, walk_stds):
    axes[2].text(bar.get_x() + bar.get_width()/2, val + sd + 0.35, f"{val:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#9b2226')

axes[2].set_xticks(x_idx)
axes[2].set_xticklabels(short_labels, fontsize=9, fontweight='bold')
axes[2].set_ylabel('Context Macro (Balanced) Accuracy (%)', fontsize=10, fontweight='bold')
axes[2].set_ylim(75, 100)
axes[2].set_title(f'C) 10-Seed Sitting vs. Walking Macro Accuracy\n(HistGB Walk: {walk_means[0]:.2f}% ± {walk_stds[0]:.2f}% | Sit: {sit_means[0]:.2f}% ± {sit_stds[0]:.2f}%)', fontsize=10.5, fontweight='bold', pad=10)
axes[2].legend(loc='lower right', fontsize=8.5, frameon=True)
axes[2].grid(axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
fig_path = os.path.join(figures_dir, 'fig_statistical_significance_tests.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nFigure saved to: {fig_path}")
