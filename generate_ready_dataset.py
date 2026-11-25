import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import numpy as np
import pandas as pd

BASE_DIR     = r"C:\adavtive_dataset"
RAW_HMOG_ACT = os.path.join(BASE_DIR, "raw_data", "hmog", "hmog_activity_features.csv")
OUT_DIR      = os.path.join(BASE_DIR, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Load raw HMOG activity features and keep writing + map tasks
df_hmog = pd.read_csv(RAW_HMOG_ACT)
hmog_clean = df_hmog[df_hmog['task_type'].isin(['writing', 'map'])].copy().reset_index(drop=True)
print(f"Loaded HMOG: {hmog_clean.shape[0]} rows, {hmog_clean['user_id'].nunique()} users")

# Fill missing typing/swipe values using each user's own median in the same context
user_ctx_writing = hmog_clean[hmog_clean['task_type'] == 'writing'].groupby(['user_id', 'context'])[['typing_speed_kpm', 'backspace_rate']].median().reset_index()
user_writing     = hmog_clean[hmog_clean['task_type'] == 'writing'].groupby('user_id')[['typing_speed_kpm', 'backspace_rate']].median().reset_index()

user_ctx_swipe = hmog_clean.dropna(subset=['swipe_speed_mean_pxps']).groupby(['user_id', 'context'])[['swipe_speed_mean_pxps', 'swipe_distance_mean_px']].median().reset_index()
user_swipe     = hmog_clean.dropna(subset=['swipe_speed_mean_pxps']).groupby('user_id')[['swipe_speed_mean_pxps', 'swipe_distance_mean_px']].median().reset_index()

hmog_clean = hmog_clean.merge(user_ctx_writing, on=['user_id', 'context'], how='left', suffixes=('', '_uctx'))
hmog_clean = hmog_clean.merge(user_writing, on='user_id', how='left', suffixes=('', '_u'))
hmog_clean = hmog_clean.merge(user_ctx_swipe, on=['user_id', 'context'], how='left', suffixes=('', '_uctx'))
hmog_clean = hmog_clean.merge(user_swipe, on='user_id', how='left', suffixes=('', '_u'))

user_med_dur = hmog_clean.groupby('user_id')['touch_duration_mean_ms'].transform('median').clip(lower=50.0)
dur_ratio = (hmog_clean['touch_duration_mean_ms'] / user_med_dur).clip(0.75, 1.35)

hmog_clean['typing_speed_kpm'] = hmog_clean['typing_speed_kpm'].fillna(hmog_clean['typing_speed_kpm_uctx'] / dur_ratio).fillna(hmog_clean['typing_speed_kpm_u'] / dur_ratio).fillna(185.0)
hmog_clean['backspace_rate']   = hmog_clean['backspace_rate'].fillna(hmog_clean['backspace_rate_uctx'] * dur_ratio).fillna(hmog_clean['backspace_rate_u'] * dur_ratio).fillna(0.085)
hmog_clean['swipe_speed_mean_pxps']  = hmog_clean['swipe_speed_mean_pxps'].fillna(hmog_clean['swipe_speed_mean_pxps_uctx'] / dur_ratio).fillna(hmog_clean['swipe_speed_mean_pxps_u']).fillna(850.0)
hmog_clean['swipe_distance_mean_px'] = hmog_clean['swipe_distance_mean_px'].fillna(hmog_clean['swipe_distance_mean_px_uctx']).fillna(hmog_clean['swipe_distance_mean_px_u']).fillna(300.0)

# Build final 17-column dataset
rng = np.random.default_rng(42)
final_rows = []

for _, row in hmog_clean.iterrows():
    walk    = 1 if str(row['context']).lower() == 'walking' else 0
    dur_ms  = float(np.clip(row['touch_duration_mean_ms'] if pd.notna(row['touch_duration_mean_ms']) else 350.0, 16.0, 2656.0))
    sw_spd  = float(np.clip(row['swipe_speed_mean_pxps'] if pd.notna(row['swipe_speed_mean_pxps']) else 850.0, 60.0, 14000.0))
    sw_dst  = float(np.clip(row['swipe_distance_mean_px'] if pd.notna(row['swipe_distance_mean_px']) else 300.0, 7.5, 1450.0))
    typ_spd = float(np.clip(row['typing_speed_kpm'] if pd.notna(row['typing_speed_kpm']) else 185.0, 50.0, 400.0))
    bk_rate = float(np.clip(row['backspace_rate'] if pd.notna(row['backspace_rate']) else 0.085, 0.0, 0.50))
    c_size  = float(np.clip(row['contact_size_mean'] if pd.notna(row['contact_size_mean']) else 0.025, 0.0175, 0.0363))
    tx      = float(np.clip(row['touch_x_mean'] if pd.notna(row['touch_x_mean']) else 600.0, 45.0, 1605.0))
    ty      = float(np.clip(row['touch_y_mean'] if pd.notna(row['touch_y_mean']) else 750.0, 12.0, 1925.0))

    # Normalize features to [0, 1]
    dur_norm     = float(np.clip((dur_ms - 120.0) / (750.0 - 120.0), 0.0, 1.0))
    contact_norm = float(np.clip((c_size - 0.019) / (0.032 - 0.019), 0.0, 1.0))
    swipe_norm   = float(np.clip((sw_spd - 500.0) / (2200.0 - 500.0), 0.0, 1.0))
    reach_norm   = float(np.clip(((tx - 650.0) / 550.0)**2 * 0.45 + ((ty - 750.0) / 650.0)**2 * 0.55, 0.0, 1.0))
    err_norm     = float(np.clip(bk_rate / 0.20, 0.0, 1.0))

    # Estimate touch miss rate from movement, contact size, dwell time, swipe, and reach
    miss_raw = (
        0.025
        + 0.052 * walk
        + 0.072 * (contact_norm ** 1.3)
        + 0.062 * (dur_norm ** 1.2)
        + 0.042 * abs(swipe_norm - 0.45)
        + 0.038 * reach_norm
        + 0.052 * err_norm
        + 0.032 * (walk * contact_norm)
    )
    miss_rate = float(np.clip(miss_raw, 0.02, 0.38))
    tap_acc   = 1.0 - miss_rate

    # Estimate reaction latency
    reaction_ms = float(np.clip(
        360.0 - 0.70 * typ_spd + 0.18 * dur_ms + 42.0 * walk + 0.06 * sw_dst + 45.0 * err_norm,
        180.0, 680.0
    ))
    reaction_norm = float(np.clip((reaction_ms - 180.0) / (620.0 - 180.0), 0.0, 1.0))

    # Combine all 5 feature domains into target button size score
    p_walk        = 1.0 if walk == 1 else 0.0
    p_touch       = (1.0 if dur_ms > 460.0 else 0.0) + (0.8 if c_size > 0.0254 else 0.0)
    p_swipe_reach = 0.7 if (sw_dst > 365.0 or ty > 920.0 or sw_spd > 1480.0 or tx > 900.0) else 0.0
    p_typing      = (1.0 if bk_rate > 0.095 else 0.0) + (0.6 if typ_spd < 174.0 else 0.0)
    p_error       = (1.2 if miss_rate > 0.165 else 0.0) + (0.8 if reaction_ms > 348.0 else 0.0)

    total_score = p_walk + p_touch + p_swipe_reach + p_typing + p_error

    if total_score >= 4.75:
        target_btn = 3  # XL (64dp)
    elif total_score >= 3.25:
        target_btn = 2  # Large (56dp)
    elif total_score >= 1.75:
        target_btn = 1  # Default (48dp)
    else:
        target_btn = 0  # Small (40dp)

    final_rows.append({
        'user_id': f"hmog_{row['user_id']}",
        'session_id': f"s_{row['activity_id']}",
        'source_dataset': 'HMOG',
        'context_is_walking': walk,
        'touch_duration_mean_ms': round(dur_ms, 2),
        'swipe_speed_mean_pxps': round(sw_spd, 2),
        'swipe_distance_mean_px': round(sw_dst, 2),
        'typing_speed_kpm': round(typ_spd, 2),
        'backspace_rate': round(bk_rate, 4),
        'contact_size_mean': round(c_size, 4),
        'touch_x_mean': round(tx, 2),
        'touch_y_mean': round(ty, 2),
        'tap_accuracy': round(tap_acc, 4),
        'miss_rate': round(miss_rate, 4),
        'reaction_time_mean_ms': round(reaction_ms, 2),
        'reaction_time_norm': round(reaction_norm, 4),
        'target_button_size': target_btn
    })

ready_df = pd.DataFrame(final_rows)

# Exact 25% per class via stratified undersampling (keeps 100% real human sessions)
min_count = ready_df['target_button_size'].value_counts().min()
ready_df = pd.concat(
    [grp.sample(min_count, random_state=42) for _, grp in ready_df.groupby('target_button_size')],
    ignore_index=True
).sample(frac=1.0, random_state=42).reset_index(drop=True)

out_csv = os.path.join(OUT_DIR, "final_ready_dataset.csv")
ready_df.to_csv(out_csv, index=False)
print(f"Saved {out_csv} ({ready_df.shape[0]} rows, {ready_df.shape[1]} cols, users={ready_df['user_id'].nunique()}, nulls={ready_df.isna().sum().sum()})")
print("Class counts:", ready_df['target_button_size'].value_counts().sort_index().to_dict())
print("Class %:", (ready_df['target_button_size'].value_counts(normalize=True).sort_index() * 100).round(2).to_dict())
