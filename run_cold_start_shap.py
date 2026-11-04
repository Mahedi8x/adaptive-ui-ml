import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = r"C:\Users\Mehedi\adavtive_model\adavtive_dataset"
DATA_PATH = os.path.join(BASE_DIR, "processed", "final_ready_dataset.csv")
FIG_DIR = os.path.join(BASE_DIR, "paper_figures", "cold_start_shap")
os.makedirs(FIG_DIR, exist_ok=True)

print("="*65)
print("  Cold-Start ML Benchmarking & Biomechanical SHAP Analysis")
print("="*65)

# 1. Load Data & Clean
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape[0]} rows, {df['user_id'].nunique()} distinct users.")

# Apply physiological bounds from EDA findings
df['touch_duration_mean_ms'] = df['touch_duration_mean_ms'].clip(15.0, 1500.0)
df['swipe_speed_mean_pxps'] = df['swipe_speed_mean_pxps'].clip(10.0, 4000.0)
df['reaction_time_mean_ms'] = df['reaction_time_mean_ms'].clip(160.0, 1200.0)
df['reaction_time_norm'] = (df['reaction_time_mean_ms'] - 160.0) / (1200.0 - 160.0)
df['miss_rate'] = df['miss_rate'].clip(0.0, 1.0)
df['tap_accuracy'] = 1.0 - df['miss_rate']

FEATURE_COLS = [
    'context_is_walking',
    'touch_duration_mean_ms',
    'swipe_speed_mean_pxps',
    'swipe_distance_mean_px',
    'contact_size_mean',
    'touch_x_mean',
    'touch_y_mean',
    'tap_accuracy',
    'miss_rate',
    'reaction_time_mean_ms',
    'reaction_time_norm'
]
TARGET_COL = 'target_button_size'
GROUP_COL = 'user_id'

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy().astype(int)
groups = df[GROUP_COL].copy()
X = X.fillna(X.median())

# 2. 5-Fold GroupKFold Cross-Validation for 5 Baselines
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Support Vector Machine (SVM)': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    'Decision Tree (CART)': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    'XGBoost (Proposed)': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, eval_metric='mlogloss', random_state=42)
}

results = []
best_model_name = None
best_f1 = 0.0
best_y_true, best_y_pred = [], []
gkf = GroupKFold(n_splits=5)

print("\nRunning 5-Fold Cross-Validation partitioned strictly by user_id...")
for name, model in models.items():
    fold_acc, fold_prec, fold_rec, fold_macro_f1, fold_weighted_f1 = [], [], [], [], []
    fold_latencies = []
    curr_y_true, curr_y_pred = [], []
    
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model.fit(X_train_scaled, y_train)
        
        t0 = time.perf_counter()
        preds = model.predict(X_val_scaled)
        latency = ((time.perf_counter() - t0) / len(X_val)) * 1000.0
        
        fold_acc.append(accuracy_score(y_val, preds))
        fold_prec.append(precision_score(y_val, preds, average='weighted', zero_division=0))
        fold_rec.append(recall_score(y_val, preds, average='weighted', zero_division=0))
        fold_macro_f1.append(f1_score(y_val, preds, average='macro', zero_division=0))
        fold_weighted_f1.append(f1_score(y_val, preds, average='weighted', zero_division=0))
        fold_latencies.append(latency)
        
        curr_y_true.extend(y_val)
        curr_y_pred.extend(preds)
        
    avg_acc = np.mean(fold_acc)
    avg_f1_w = np.mean(fold_weighted_f1)
    avg_f1_m = np.mean(fold_macro_f1)
    avg_lat = np.mean(fold_latencies)
    
    results.append({
        'Algorithm': name,
        'Accuracy (%)': round(avg_acc * 100, 2),
        'Precision (%)': round(np.mean(fold_prec) * 100, 2),
        'Recall (%)': round(np.mean(fold_rec) * 100, 2),
        'Macro F1 (%)': round(avg_f1_m * 100, 2),
        'Weighted F1 (%)': round(avg_f1_w * 100, 2),
        'Latency (ms/sample)': round(avg_lat, 4)
    })
    print(f"  {name:30s} | Acc: {avg_acc*100:6.2f}% | F1: {avg_f1_w*100:6.2f}% | Latency: {avg_lat:6.4f} ms")
    
    if avg_f1_w > best_f1:
        best_f1 = avg_f1_w
        best_model_name = name
        best_y_true = curr_y_true
        best_y_pred = curr_y_pred

table2_df = pd.DataFrame(results).sort_values(by='Weighted F1 (%)', ascending=False).reset_index(drop=True)
table2_path = os.path.join(FIG_DIR, "table2_baseline_comparison.csv")
table2_df.to_csv(table2_path, index=False)
print(f"\n[Saved Table 2] Benchmark results saved to: {table2_path}")

# 3. Figure 6: Baseline Performance Comparison Barplot
plt.figure(figsize=(10, 5))
x_pos = np.arange(len(table2_df))
width = 0.35
plt.bar(x_pos - width/2, table2_df['Accuracy (%)'], width, label='Accuracy (%)', color='#2b5c8f', edgecolor='black')
plt.bar(x_pos + width/2, table2_df['Weighted F1 (%)'], width, label='Weighted F1 (%)', color='#38a3a5', edgecolor='black')
plt.xticks(x_pos, table2_df['Algorithm'], rotation=15, ha='right', fontsize=10)
plt.ylabel('Percentage (%)', fontsize=11)
plt.title('Cold-Start Classification Performance across 5 Model Families\n(Strict 5-Fold GroupKFold by User ID)', fontsize=13, pad=15)
plt.ylim(85, 105)
plt.legend(loc='lower right', fontsize=10)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
fig6_path = os.path.join(FIG_DIR, "fig6_baseline_model_comparison_barchart.png")
plt.savefig(fig6_path, dpi=300)
plt.close()
print(f"[Saved Fig 6] Comparison Bar Chart: {fig6_path}")

# 4. Figure 7: Confusion Matrix for Champion Model
plt.figure(figsize=(7, 6))
labels = ['Small (40dp)', 'Default (48dp)', 'Large (56dp)', 'Extra Large (64dp)']
cm = confusion_matrix(best_y_true, best_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)
plt.title(f'Confusion Matrix: {best_model_name}\n(Cross-User Evaluation, Zero Data Leakage)', fontsize=12, pad=12)
plt.xlabel('Predicted Optimal Target Size', fontsize=11)
plt.ylabel('Ground Truth Target Size', fontsize=11)
plt.tight_layout()
fig7_path = os.path.join(FIG_DIR, "fig7_champion_model_confusion_matrix.png")
plt.savefig(fig7_path, dpi=300)
plt.close()
print(f"[Saved Fig 7] Confusion Matrix: {fig7_path}")

# 5. Fit Final XGBoost Model for SHAP Explainability
print("\nFitting Champion XGBoost model on full dataset for SHAP explainability...")
xgb_final = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, eval_metric='mlogloss', random_state=42)
scaler_final = StandardScaler()
X_scaled = pd.DataFrame(scaler_final.fit_transform(X), columns=FEATURE_COLS)
xgb_final.fit(X_scaled, y)

# 6. SHAP Calculations with TreeExplainer
print("Computing SHAP values via TreeExplainer...")
explainer = shap.TreeExplainer(xgb_final)
shap_values = explainer.shap_values(X_scaled)

# 7. Figure 8: Global SHAP Feature Importance Bar Plot
plt.figure(figsize=(10, 6))
# For multi-class, mean absolute SHAP over classes or directly via shap.summary_plot
shap.summary_plot(shap_values, X, plot_type="bar", class_names=labels, show=False)
plt.title('Global Biomechanical Feature Importance (Mean |SHAP Value| Across Classes)', fontsize=13, pad=15)
plt.xlabel('Mean |SHAP Value| (Impact on Model Output)', fontsize=11)
plt.tight_layout()
fig8_path = os.path.join(FIG_DIR, "fig8_shap_global_feature_importance_bar.png")
plt.savefig(fig8_path, dpi=300)
plt.close()
print(f"[Saved Fig 8] SHAP Global Feature Importance: {fig8_path}")

# 8. Figure 9: SHAP Beeswarm Plot for Class 2 (Large Button UI)
# Demonstrates directional impact: higher miss_rate -> increases log-odds of Large Button
plt.figure(figsize=(10, 6))
# Handle list of matrices for multiclass vs 3D array
if isinstance(shap_values, list):
    shap_val_large = shap_values[2] # Class 2 = Large
else:
    shap_val_large = shap_values[:, :, 2] if len(shap_values.shape) == 3 else shap_values

shap.summary_plot(shap_val_large, X, show=False)
plt.title('SHAP Directional Impact for Large Button UI (Class 2: 56dp)\n(Red = High Feature Value, Blue = Low Feature Value)', fontsize=12, pad=15)
plt.tight_layout()
fig9_path = os.path.join(FIG_DIR, "fig9_shap_beeswarm_directional_plot.png")
plt.savefig(fig9_path, dpi=300)
plt.close()
print(f"[Saved Fig 9] SHAP Directional Beeswarm Plot: {fig9_path}")

# 9. Figure 10: Individual User Sample Decision (Waterfall / Force explanation)
sample_idx = 10 # Example walking user with high miss rate
plt.figure(figsize=(9, 5))
# Using single class shap values for sample
sample_shap = shap_val_large[sample_idx]
sample_x = X.iloc[sample_idx]
top_n = 7
sorted_idx = np.argsort(np.abs(sample_shap))[-top_n:]
plt.barh(np.arange(top_n), sample_shap[sorted_idx], color=np.where(sample_shap[sorted_idx] > 0, '#e74c3c', '#2ecc71'), edgecolor='black')
plt.yticks(np.arange(top_n), [f"{sample_x.index[i]} = {sample_x.iloc[i]:.2f}" for i in sorted_idx], fontsize=10)
plt.xlabel('SHAP Value (Contribution to Large Button Recommendation)', fontsize=11)
plt.title(f'Local Interpretability for Subject #{df.iloc[sample_idx]["user_id"]}\n(Positive SHAP pushes model toward larger target size)', fontsize=12, pad=15)
plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
fig10_path = os.path.join(FIG_DIR, "fig10_shap_waterfall_individual_sample.png")
plt.savefig(fig10_path, dpi=300)
plt.close()
print(f"[Saved Fig 10] Individual Sample SHAP Explanation: {fig10_path}")

print("\nAll Cold-Start Benchmarks and SHAP Visualizations generated successfully!")
