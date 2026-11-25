import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score

train_path = os.path.join(os.path.dirname(__file__), "processed", "final_ready_dataset.csv")
test_path  = os.path.join(os.path.dirname(__file__), "processed", "bbmas_test_ready_dataset.csv")
figures_dir = os.path.join(os.path.dirname(__file__), "paper_figures", "cold_start_shap")
os.makedirs(figures_dir, exist_ok=True)

df_train = pd.read_csv(train_path)
df_test  = pd.read_csv(test_path)

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

X = df_train[FEATURE_COLS].copy()
y = df_train[TARGET_COL].copy().astype(int)

X_ext = df_test[FEATURE_COLS].copy()
y_ext = df_test[TARGET_COL].copy().astype(int)

def get_candidate_models():
    return {
        'HistGradientBoosting': HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.08, random_state=42
        ),
        'Hybrid Ensemble (XGB+LGBM+Cat)': VotingClassifier([
            ('xgb', XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, eval_metric='mlogloss', random_state=42)),
            ('lgbm', LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, random_state=42, verbose=-1)),
            ('cat', CatBoostClassifier(iterations=300, depth=6, learning_rate=0.08, random_seed=42, verbose=0))
        ], voting='soft'),
        'LightGBM': LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.08, random_state=42, verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.08, random_seed=42, verbose=0
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.08, eval_metric='mlogloss', random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=180, max_depth=10, random_state=42
        )
    }

print("Running 5-Step Random Shuffle-Split Cross-Validation Stability Test...")
print("=" * 70)
ss = ShuffleSplit(n_splits=5, test_size=0.20, random_state=42)

step_records = []

for step_idx, (train_idx, val_idx) in enumerate(ss.split(X, y), 1):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

    scaler = RobustScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_va_s  = scaler.transform(X_va)
    X_ext_s = scaler.transform(X_ext)

    models = get_candidate_models()
    for model_name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_tr_s, y_tr)
        fit_time = time.perf_counter() - t0

        val_preds = model.predict(X_va_s)
        ext_preds = model.predict(X_ext_s)

        val_acc = accuracy_score(y_va, val_preds) * 100.0
        ext_acc = accuracy_score(y_ext, ext_preds) * 100.0
        ext_f1  = f1_score(y_ext, ext_preds, average='weighted', zero_division=0) * 100.0

        step_records.append({
            'Step': f"Step {step_idx}",
            'Model': model_name,
            'Train Samples': len(X_tr),
            'Val Samples': len(X_va),
            'Val Acc (%)': round(val_acc, 2),
            'BB-MAS Test Acc (%)': round(ext_acc, 2),
            'BB-MAS Weighted F1 (%)': round(ext_f1, 2)
        })

df_steps = pd.DataFrame(step_records)

# Summary table: Mean ± Std
summary_rows = []
for model_name in df_steps['Model'].unique():
    m_df = df_steps[df_steps['Model'] == model_name]
    val_m, val_s = m_df['Val Acc (%)'].mean(), m_df['Val Acc (%)'].std()
    ext_m, ext_s = m_df['BB-MAS Test Acc (%)'].mean(), m_df['BB-MAS Test Acc (%)'].std()
    f1_m, f1_s   = m_df['BB-MAS Weighted F1 (%)'].mean(), m_df['BB-MAS Weighted F1 (%)'].std()

    summary_rows.append({
        'Model': model_name,
        'Val Acc Mean (%)': round(val_m, 2),
        'Val Acc Std': round(val_s, 2),
        'BB-MAS Test Acc Mean (%)': round(ext_m, 2),
        'BB-MAS Test Acc Std': round(ext_s, 2),
        'BB-MAS Test F1 Mean (%)': round(f1_m, 2),
        'BB-MAS Test F1 Std': round(f1_s, 2),
        'Summary (Test Acc)': f"{ext_m:.2f} ± {ext_s:.2f}%"
    })

df_summary = pd.DataFrame(summary_rows).sort_values(by='BB-MAS Test Acc Mean (%)', ascending=False).reset_index(drop=True)
summary_csv = os.path.join(figures_dir, 'table_shuffle_split_stability.csv')
df_summary.to_csv(summary_csv, index=False)

print("\n--- 5-Step Shuffle-Split Cross-Validation Stability Summary ---")
print(df_summary[['Model', 'Val Acc Mean (%)', 'Val Acc Std', 'BB-MAS Test Acc Mean (%)', 'BB-MAS Test Acc Std', 'Summary (Test Acc)']].to_string(index=False))

# Visualization
plt.figure(figsize=(12, 5.5))
palette = ['#1d3557', '#457b9d', '#2a9d8f', '#e76f51', '#f4a261', '#6c757d']
sns.barplot(
    data=df_steps,
    x='Step',
    y='BB-MAS Test Acc (%)',
    hue='Model',
    palette=palette,
    edgecolor='black',
    alpha=0.9
)
plt.title("5-Step Random Shuffle-Split Stability on Unseen BB-MAS Users\n(Evaluating Robustness Across Random 80/20 Internal Partitions)", fontsize=12, pad=12, fontweight='bold')
plt.xlabel("Shuffle Split Run", fontsize=11, fontweight='bold')
plt.ylabel("External Test Accuracy (%)", fontsize=11, fontweight='bold')
plt.ylim(80, 95)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=9.5)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
fig_path = os.path.join(figures_dir, 'fig_shuffle_split_stability.png')
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"\nFigure saved to: {fig_path}")
print(f"Summary table saved to: {summary_csv}")
