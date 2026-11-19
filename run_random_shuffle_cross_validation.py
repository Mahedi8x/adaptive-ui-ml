import os
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

# ১. ডেটাসেট লোড করা (HMOG এবং BB-MAS)
train_path = r"C:\adavtive_dataset\processed\final_ready_dataset.csv"
test_path  = r"C:\adavtive_dataset\processed\bbmas_test_ready_dataset.csv"

df_train = pd.read_csv(train_path)
df_test  = pd.read_csv(test_path)

# ২. ফিচার এবং টার্গেট কলাম নির্ধারণ
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

print(f"Total Rows in HMOG: {len(X)}")

# ৩. ৫-ধাপের র‍্যান্ডম শাফল স্প্লিট (Random Shuffle-Split)
ss = ShuffleSplit(n_splits=5, test_size=0.20, random_state=42)

step_results = []
step_idx = 1

for train_idx, val_idx in ss.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # RobustScaler দিয়ে স্কেলিং
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_ext_s = scaler.transform(X_ext)
    
    # XGBoost মডেল ট্রেইন
    xgb = XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.08, 
        eval_metric='mlogloss', 
        random_state=42
    )
    xgb.fit(X_tr_s, y_tr)
    
    # ভ্যালিডেশন প্রেডিকশন
    va_preds = xgb.predict(X_va_s)
    va_acc = accuracy_score(y_va, va_preds)
    
    # এক্সটার্নাল টেস্ট প্রেডিকশন (BB-MAS)
    ext_preds = xgb.predict(X_ext_s)
    ext_acc = accuracy_score(y_ext, ext_preds)
    ext_f1  = f1_score(y_ext, ext_preds, average='weighted', zero_division=0)
    
    step_results.append({
        'Round / Step': f"Step {step_idx}",
        'Random Train Size': len(X_tr),
        'Random Val Size': len(X_va),
        'Internal Val Acc (%)': round(va_acc * 100, 2),
        'BB-MAS External Acc (%)': round(ext_acc * 100, 2),
        'BB-MAS Weighted F1 (%)': round(ext_f1 * 100, 2)
    })
    step_idx += 1

# ৪. ফলাফল প্রদর্শন
res_df = pd.DataFrame(step_results)
print("\n--- 5-Step Random Shuffle Cross-Validation Results ---")
print(res_df.to_string(index=False))

avg_va = res_df['Internal Val Acc (%)'].mean()
avg_ext = res_df['BB-MAS External Acc (%)'].mean()
print(f"\nAverage Internal Validation Accuracy: {avg_va:.2f}%")
print(f"Average External BB-MAS Test Accuracy: {avg_ext:.2f}%")
