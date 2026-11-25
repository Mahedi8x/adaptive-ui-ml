import os
import time
import warnings
warnings.filterwarnings('ignore')
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

base_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, "processed", "final_ready_dataset.csv")
test_path  = os.path.join(base_dir, "processed", "bbmas_test_ready_dataset.csv")
figures_dir = os.path.join(base_dir, "paper_figures", "cold_start_shap")
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

X_train = df_train[FEATURE_COLS].copy()
y_train = df_train[TARGET_COL].copy().astype(int)

X_test = df_test[FEATURE_COLS].copy()
y_test = df_test[TARGET_COL].copy().astype(int)

def build_shuffle_models():
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

ss = ShuffleSplit(n_splits=5, test_size=0.20, random_state=42)
step_records = []

for step_idx, (train_idx, val_idx) in enumerate(ss.split(X_train, y_train), 1):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

    scaler = RobustScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_va_s  = scaler.transform(X_va)
    X_ext_s = scaler.transform(X_test)

    models = build_shuffle_models()
    for model_name, model in models.items():
        model.fit(X_tr_s, y_tr)
        va_preds  = model.predict(X_va_s)
        ext_preds = model.predict(X_ext_s)

        va_acc  = accuracy_score(y_va, va_preds) * 100.0
        va_f1   = f1_score(y_va, va_preds, average='weighted', zero_division=0) * 100.0
        ext_acc = accuracy_score(y_test, ext_preds) * 100.0
        ext_f1  = f1_score(y_test, ext_preds, average='weighted', zero_division=0) * 100.0

        step_records.append({
            'Step': f"Step {step_idx}",
            'Model': model_name,
            'Train Size': len(X_tr),
            'Val Size': len(X_va),
            'Val Acc (%)': round(va_acc, 2),
            'Val F1 (%)': round(va_f1, 2),
            'External Test Acc (%)': round(ext_acc, 2),
            'External Test F1 (%)': round(ext_f1, 2)
        })

df_steps = pd.DataFrame(step_records)

# 1. Step-by-Step Scenario Table (All 5 Steps across All Models - External Test Acc)
model_order = [
    'HistGradientBoosting',
    'Hybrid Ensemble (XGB+LGBM+Cat)',
    'LightGBM',
    'XGBoost',
    'CatBoost',
    'Random Forest'
]

step_test_pivot = df_steps.pivot(index='Step', columns='Model', values='External Test Acc (%)')[model_order].reset_index()
step_test_pivot.insert(1, 'HMOG Train (80%)', 4588)
step_test_pivot.insert(2, 'HMOG Val (20%)', 1148)

step_val_pivot = df_steps.pivot(index='Step', columns='Model', values='Val Acc (%)')[model_order].reset_index()
step_val_pivot.insert(1, 'HMOG Train (80%)', 4588)
step_val_pivot.insert(2, 'HMOG Val (20%)', 1148)

print("--- 5-Step Shuffle Scenarios: Unseen BB-MAS Test Accuracy (%) [Trained Only on HMOG] ---")
print(step_test_pivot.to_string(index=False))

print("\n--- 5-Step Shuffle Scenarios: Internal HMOG Validation Accuracy (%) ---")
print(step_val_pivot.to_string(index=False))

# 2. Overall 5-Step Average Summary Table
summary_rows = []
for model_name in df_steps['Model'].unique():
    m_df = df_steps[df_steps['Model'] == model_name]
    val_m, val_s = m_df['Val Acc (%)'].mean(), m_df['Val Acc (%)'].std()
    ext_m, ext_s = m_df['External Test Acc (%)'].mean(), m_df['External Test Acc (%)'].std()
    f1_m, f1_s   = m_df['External Test F1 (%)'].mean(), m_df['External Test F1 (%)'].std()

    summary_rows.append({
        'Model': model_name,
        'Avg Val Acc (%)': round(val_m, 2),
        'Val Std (%)': round(val_s, 2),
        'Avg External Test Acc (%)': round(ext_m, 2),
        'Test Std (%)': round(ext_s, 2),
        'Avg External Test F1 (%)': round(f1_m, 2),
        '5-Step Test Stability': f"{ext_m:.2f} ± {ext_s:.2f}%"
    })

df_summary = pd.DataFrame(summary_rows).sort_values(
    by=['Avg External Test Acc (%)', 'Avg Val Acc (%)'], ascending=False
).reset_index(drop=True)
df_summary.insert(0, 'Rank', range(1, len(df_summary) + 1))

print("\n--- 5-Step Random Shuffle-Split Average & Ranking Summary ---")
print(df_summary.to_string(index=False))

# 3. Winner Model Step-by-Step Breakdown (Classic Format)
winner_name = df_summary.loc[0, 'Model']
winner_steps = df_steps[df_steps['Model'] == winner_name][
    ['Step', 'Train Size', 'Val Size', 'Val Acc (%)', 'External Test Acc (%)', 'External Test F1 (%)']
].reset_index(drop=True)

print(f"\n--- Winning Model ({winner_name}) Detailed 5-Step Breakdown ---")
print(winner_steps.to_string(index=False))

print("\n" + "=" * 75)
print(f"WINNING MODEL (5-Step Shuffle-Split Average): {winner_name}")
print(f"-> Average Internal Validation Accuracy across 5 random steps: {df_summary.loc[0, 'Avg Val Acc (%)']:.2f}% (±{df_summary.loc[0, 'Val Std (%)']:.2f}%)")
print(f"-> Average External BB-MAS Test Accuracy across 5 random steps: {df_summary.loc[0, 'Avg External Test Acc (%)']:.2f}% (±{df_summary.loc[0, 'Test Std (%)']:.2f}%)")
print(f"-> Average External BB-MAS Test F1-Score across 5 random steps: {df_summary.loc[0, 'Avg External Test F1 (%)']:.2f}%")
print("=" * 75)

# Save CSVs
df_summary.to_csv(os.path.join(figures_dir, 'table_shuffle_split_stability.csv'), index=False)
step_test_pivot.to_csv(os.path.join(figures_dir, 'table_shuffle_split_stepwise_test.csv'), index=False)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

# Left subplot: All models across 5 steps
palette = ['#1d3557', '#457b9d', '#2a9d8f', '#e76f51', '#f4a261', '#6c757d']
sns.barplot(
    data=df_steps,
    x='Step',
    y='External Test Acc (%)',
    hue='Model',
    hue_order=model_order,
    palette=palette,
    edgecolor='black',
    alpha=0.9,
    ax=axes[0]
)
axes[0].set_title("A) All Models Across 5 Random Shuffle Scenarios\n(BB-MAS Unseen Users External Test Accuracy)", fontsize=11.5, fontweight='bold', pad=10)
axes[0].set_xlabel("Shuffle Scenario", fontsize=10.5, fontweight='bold')
axes[0].set_ylabel("External Test Accuracy (%)", fontsize=10.5, fontweight='bold')
axes[0].set_ylim(82, 94)
axes[0].legend(loc='lower right', frameon=True, fontsize=8.5)
axes[0].grid(axis='y', linestyle=':', alpha=0.6)

# Right subplot: Winning model Internal Val vs External Test across 5 steps
x_pos = np.arange(len(winner_steps))
w = 0.35
axes[1].bar(x_pos - w/2, winner_steps['Val Acc (%)'], w, label='Internal Val Acc (%)', color='#2b5c8f', edgecolor='black')
axes[1].bar(x_pos + w/2, winner_steps['External Test Acc (%)'], w, label='BB-MAS External Test Acc (%)', color='#e76f51', edgecolor='black')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(winner_steps['Step'], fontsize=10.5, fontweight='bold')
axes[1].set_ylabel("Accuracy (%)", fontsize=10.5, fontweight='bold')
axes[1].set_title(f"B) Winning Model ({winner_name})\n5-Step Internal Val vs. External Test Stability", fontsize=11.5, fontweight='bold', pad=10)
axes[1].set_ylim(85, 100)
axes[1].legend(loc='lower right', frameon=True, fontsize=9.5)
axes[1].grid(axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
fig_path = os.path.join(figures_dir, 'fig_shuffle_split_stability.png')
plt.savefig(fig_path, dpi=300)
plt.close()
