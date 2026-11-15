import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier
import joblib

# Paths
BASE_DIR = r"C:\Users\Mehedi\adavtive_model"
DATA_PATH = os.path.join(BASE_DIR, "adavtive_dataset", "processed", "final_ready_dataset.csv")
OUT_DIR = os.path.join(BASE_DIR, "member1_deliverables")
os.makedirs(OUT_DIR, exist_ok=True)

print("="*60)
print("  Member 1: Training Cold-Start ML Model (GroupKFold)")
print("="*60)

# 1. Load Data
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]} rows, {df['user_id'].nunique()} unique users.")

# Feature columns
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
y = df[TARGET_COL].copy()
groups = df[GROUP_COL].copy()

# Fill any residual NaNs with median values
X = X.fillna(X.median())

# 2. Evaluate with GroupKFold (5-Fold by User)
gkf = GroupKFold(n_splits=5)
xgb_accuracies = []
xgb_f1s = []

print("\nRunning 5-Fold Cross Validation by User ID...")
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='weighted')
    xgb_accuracies.append(acc)
    xgb_f1s.append(f1)
    print(f"  Fold {fold}: Accuracy = {acc*100:.2f}%, Weighted F1 = {f1*100:.2f}%")

mean_acc = np.mean(xgb_accuracies)
mean_f1 = np.mean(xgb_f1s)
print(f"\n---> Average Cross-Validation Accuracy: {mean_acc*100:.2f}%")
print(f"---> Average Cross-Validation F1-Score: {mean_f1*100:.2f}%")

# 3. Train Final Model on All Data & Export
final_xgb = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.08,
    random_state=42,
    eval_metric='mlogloss'
)
final_xgb.fit(X, y)

# Save Model files
pkl_path = os.path.join(OUT_DIR, "cold_start_xgboost.pkl")
json_path = os.path.join(OUT_DIR, "cold_start_xgboost.json")
joblib.dump(final_xgb, pkl_path)
final_xgb.save_model(json_path)
print(f"\n[Export] Saved PKL Model: {pkl_path}")
print(f"[Export] Saved JSON Model (For Mobile/OS integration): {json_path}")

# Feature list metadata
with open(os.path.join(OUT_DIR, "model_features.json"), "w") as f:
    json.dump({
        "features": FEATURE_COLS,
        "classes": {0: "Small (40dp)", 1: "Default (48dp)", 2: "Large (56dp)", 3: "Extra Large (64dp)"},
        "metrics": {"cv_accuracy": float(mean_acc), "cv_f1": float(mean_f1)}
    }, f, indent=4)

# 4. Feature Importance Plot (For Paper)
plt.figure(figsize=(10, 6))
feat_imp = pd.Series(final_xgb.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
feat_imp.plot(kind='barh', color='#1f77b4', edgecolor='black')
plt.title('Cold-Start XGBoost Feature Importance (Adaptive UI Framework)', fontsize=14, pad=15)
plt.xlabel('F-Score / Importance Weight', fontsize=12)
plt.ylabel('User Behavior Features', fontsize=12)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "feature_importance.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"[Paper Graph] Saved Feature Importance: {fig_path}")

print("\n Cold-Start Model Training Completed Successfully!")
