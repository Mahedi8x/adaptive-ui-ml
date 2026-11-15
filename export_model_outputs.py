import os
import sys
import json
import joblib

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = r"C:\Users\Mehedi\adavtive_model\adavtive_dataset"
DATA_PATH = os.path.join(BASE_DIR, "processed", "final_ready_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
OUT_DIR = os.path.join(BASE_DIR, "model_outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

print("="*65)
print("  Generating Explicit Train-Test Splits, Predictions & Model Files")
print("="*65)

# 1. Load Data
df = pd.read_csv(DATA_PATH)
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

X = df[FEATURE_COLS].copy().fillna(df[FEATURE_COLS].median())
y = df[TARGET_COL].copy().astype(int)
groups = df[GROUP_COL].copy()

# 2. Detailed 5-Fold GroupKFold Breakdown
gkf = GroupKFold(n_splits=5)
fold_stats = []
for f, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
    tr_users = groups.iloc[tr].nunique()
    te_users = groups.iloc[te].nunique()
    fold_stats.append({
        'Fold': f'Fold {f}',
        'Train_Samples': len(tr),
        'Train_Users': tr_users,
        'Test_Samples': len(te),
        'Test_Users': te_users,
        'Test_Ratio_Samples (%)': round(len(te) / len(df) * 100, 2),
        'Test_Ratio_Users (%)': round(te_users / groups.nunique() * 100, 2)
    })

fold_stats_df = pd.DataFrame(fold_stats)
fold_stats_path = os.path.join(OUT_DIR, "01_five_fold_train_test_split_breakdown.csv")
fold_stats_df.to_csv(fold_stats_path, index=False)
print("\n[Split Breakdown] 5-Fold GroupKFold Partitioning (Partitioned by User ID):")
print(fold_stats_df.to_string(index=False))

# 3. Explicit 80/20 Holdout Test Evaluation
# (Train on 112 users, Test strictly on remaining 29 unseen users)
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
test_meta = df.iloc[test_idx][['user_id', 'session_id', 'source_dataset', 'context_is_walking', 'miss_rate', 'reaction_time_mean_ms']].copy()

print(f"\n--- Explicit Holdout Train/Test Split ---")
print(f"Training Set : {len(X_train)} samples across {groups.iloc[train_idx].nunique()} unique users ({len(X_train)/len(df)*100:.1f}%)")
print(f"Testing Set  : {len(X_test)} samples across {groups.iloc[test_idx].nunique()} unseen users ({len(X_test)/len(df)*100:.1f}%)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save Scaler
joblib.dump(scaler, os.path.join(MODELS_DIR, "feature_scaler.pkl"))

# Train all 5 models on the 80% train split
models = {
    'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision_Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Support_Vector_Machine': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    'Random_Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, eval_metric='mlogloss', random_state=42)
}

test_predictions_df = test_meta.copy()
test_predictions_df['ground_truth_target'] = y_test.values
class_map = {0: 'Small (40dp)', 1: 'Default (48dp)', 2: 'Large (56dp)', 3: 'Extra Large (64dp)'}
test_predictions_df['ground_truth_label_name'] = test_predictions_df['ground_truth_target'].map(class_map)

reports = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    
    # Save Model Artifact
    pkl_file = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, pkl_file)
    if name == 'XGBoost':
        json_file = os.path.join(MODELS_DIR, "xgboost_champion.json")
        model.save_model(json_file)
        
    test_predictions_df[f'pred_{name}'] = preds
    test_predictions_df[f'pred_name_{name}'] = pd.Series(preds).map(class_map).values
    
    # Classification Report Dictionary
    rep = classification_report(y_test, preds, target_names=['Small (40dp)', 'Default (48dp)', 'Large (56dp)', 'Extra Large (64dp)'], output_dict=True)
    rep_df = pd.DataFrame(rep).transpose().reset_index().rename(columns={'index': 'class_or_metric'})
    rep_df['model'] = name
    reports[name] = rep_df
    
    print(f"✓ Model: {name:25s} | Test Accuracy on Unseen Users: {acc*100:.2f}% | Model Saved: {pkl_file}")

# 4. Save Predictions CSV (Row-by-row proof of what was predicted)
pred_path = os.path.join(OUT_DIR, "02_test_set_predictions_row_by_row.csv")
test_predictions_df.to_csv(pred_path, index=False)
print(f"\n[Saved Output] Row-by-Row Test Set Predictions: {pred_path}")
print(f"               (Contains {len(test_predictions_df)} test rows with Ground Truth vs 5 Model Predictions)")

# 5. Save Classification Report per Class Table
all_reports_df = pd.concat(reports.values(), ignore_index=True)
rep_path = os.path.join(OUT_DIR, "03_classification_report_per_class.csv")
all_reports_df.to_csv(rep_path, index=False)
print(f"[Saved Output] Full Classification Report per Class: {rep_path}")

# Print sample of predictions
print("\n--- Sample of Predictions on Unseen Test Users (First 5 Rows) ---")
cols_to_show = ['user_id', 'context_is_walking', 'miss_rate', 'ground_truth_label_name', 'pred_name_Random_Forest', 'pred_name_XGBoost']
print(test_predictions_df[cols_to_show].head(5).to_string(index=False))

print("\nAll model artifacts and outputs exported successfully!")
