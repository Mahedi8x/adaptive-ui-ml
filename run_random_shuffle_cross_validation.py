import os
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

train_path = r"C:\adavtive_dataset\processed\final_ready_dataset.csv"
test_path  = r"C:\adavtive_dataset\processed\bbmas_test_ready_dataset.csv"

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

# 5 random 80/20 splits across the HMOG dataset
ss = ShuffleSplit(n_splits=5, test_size=0.20, random_state=42)
step_results = []

for step_idx, (train_idx, val_idx) in enumerate(ss.split(X, y), 1):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

    scaler = RobustScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_va_s  = scaler.transform(X_va)
    X_ext_s = scaler.transform(X_ext)

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        eval_metric='mlogloss',
        random_state=42
    )
    xgb.fit(X_tr_s, y_tr)

    va_acc = accuracy_score(y_va, xgb.predict(X_va_s))
    ext_preds = xgb.predict(X_ext_s)
    ext_acc = accuracy_score(y_ext, ext_preds)
    ext_f1  = f1_score(y_ext, ext_preds, average='weighted', zero_division=0)

    step_results.append({
        'Step': f"Step {step_idx}",
        'Train Size': len(X_tr),
        'Val Size': len(X_va),
        'Val Acc (%)': round(va_acc * 100, 2),
        'BB-MAS Test Acc (%)': round(ext_acc * 100, 2),
        'BB-MAS F1 (%)': round(ext_f1 * 100, 2)
    })

res_df = pd.DataFrame(step_results)
print(res_df.to_string(index=False))
print(f"\nAvg Val Acc: {res_df['Val Acc (%)'].mean():.2f}%")
print(f"Avg BB-MAS Test Acc: {res_df['BB-MAS Test Acc (%)'].mean():.2f}%")
