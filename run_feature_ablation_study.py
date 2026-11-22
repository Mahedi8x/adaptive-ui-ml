import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = r"C:\adavtive_dataset"
train_path = os.path.join(BASE_DIR, "processed", "final_ready_dataset.csv")
test_path  = os.path.join(BASE_DIR, "processed", "bbmas_test_ready_dataset.csv")

df_train = pd.read_csv(train_path)
df_test  = pd.read_csv(test_path)

y_train = df_train['target_button_size'].astype(int)
y_test  = df_test['target_button_size'].astype(int)

# Define 4 distinct feature scenarios
feature_scenarios = {
    "Scenario 1: Pure Motor Dynamics (No miss_rate)": [
        'context_is_walking',
        'touch_duration_mean_ms',
        'swipe_speed_mean_pxps',
        'swipe_distance_mean_px',
        'typing_speed_kpm',
        'backspace_rate',
        'contact_size_mean',
        'touch_x_mean',
        'touch_y_mean'
    ],
    "Scenario 2: Minimal Touch Only (No typing/error)": [
        'context_is_walking',
        'touch_duration_mean_ms',
        'swipe_speed_mean_pxps',
        'contact_size_mean',
        'touch_x_mean',
        'touch_y_mean'
    ],
    "Scenario 3: Keystroke Behavior Only": [
        'context_is_walking',
        'typing_speed_kpm',
        'backspace_rate'
    ],
    "Scenario 4: All 13 Biomechanical Features": [
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
}

results = []

for sc_name, features in feature_scenarios.items():
    X_tr = df_train[features].copy()
    X_te = df_test[features].copy()
    
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, eval_metric='mlogloss', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', C=1.0, random_state=42)
    }
    
    for m_name, model in models.items():
        model.fit(X_tr_s, y_train)
        
        tr_acc = accuracy_score(y_train, model.predict(X_tr_s))
        te_preds = model.predict(X_te_s)
        te_acc = accuracy_score(y_test, te_preds)
        te_f1  = f1_score(y_test, te_preds, average='weighted', zero_division=0)
        
        results.append({
            'Feature Scenario': sc_name,
            'Num Features': len(features),
            'Model': m_name,
            'HMOG Train Acc (%)': round(tr_acc * 100, 2),
            'BB-MAS Test Acc (%)': round(te_acc * 100, 2),
            'BB-MAS Weighted F1 (%)': round(te_f1 * 100, 2)
        })

res_df = pd.DataFrame(results)

# Save CSV Table
csv_out = os.path.join(BASE_DIR, "paper_figures", "feature_ablation_experiment_results.csv")
res_df.to_csv(csv_out, index=False)
print(f"[OK] Saved ablation table to: {csv_out}")

# Plot comparative figure
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(12, 6))

pivot_df = res_df.pivot(index='Feature Scenario', columns='Model', values='BB-MAS Test Acc (%)')
pivot_df.plot(kind='bar', ax=ax, edgecolor='black', width=0.75, colormap='Spectral')

ax.set_title('Feature Ablation Study: External BB-MAS Test Accuracy Across Feature Subsets', fontsize=12, pad=15, fontweight='bold')
ax.set_ylabel('External Test Accuracy (%)', fontsize=11, fontweight='bold')
ax.set_xlabel('')
ax.set_xticklabels(pivot_df.index, rotation=15, ha='right', fontsize=9.5, fontweight='bold')
ax.set_ylim(0, 110)
ax.axhline(25.0, color='red', linestyle='--', label='Random Chance Baseline (25%)')
ax.legend(title='Model', loc='upper right', frameon=True)
ax.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()

fig_out = os.path.join(BASE_DIR, "paper_figures", "fig_feature_ablation_comparison.png")
plt.savefig(fig_out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved ablation comparison chart to: {fig_out}")
