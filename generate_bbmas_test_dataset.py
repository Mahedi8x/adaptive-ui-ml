import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import numpy as np
import pandas as pd

# Paths
BBMAS_RAW_ROOT = r"c:\Users\Mehedi\OneDrive\Attachments\Desktop\reseach_data\BB-MAS_Dataset\BB-MAS_Dataset"
BASE_DIR       = r"C:\adavtive_dataset"
RAW_BBMAS_DIR  = os.path.join(BASE_DIR, "raw_data", "bbmas")
RAW_BBMAS_CSV  = os.path.join(RAW_BBMAS_DIR, "bbmas_activity_features.csv")
OUT_DIR        = os.path.join(BASE_DIR, "processed")

os.makedirs(RAW_BBMAS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

print("=================================================================")
print("  Processing BB-MAS Raw Data into Clean Multi-Feature Test Set")
print("  (100% Identical Multi-Feature Formula as generate_ready_dataset.py)")
print("=================================================================")

if os.path.exists(RAW_BBMAS_CSV):
    print(f"\n[1/2] Loading Extracted BB-MAS Activity Features: {RAW_BBMAS_CSV}")
    raw_bbmas_df = pd.read_csv(RAW_BBMAS_CSV)
else:
    print(f"\n[1/2] Extracting BB-MAS Raw Sensor Logs from: {BBMAS_RAW_ROOT}")
    def parse_bbmas_time(s):
        if not isinstance(s, str) or s == "null" or not s.strip():
            return pd.NaT
        if s.count(":") == 3:
            idx = s.rfind(":")
            s = s[:idx] + "." + s[idx+1:]
        return pd.to_datetime(s, errors="coerce")

    user_dirs = sorted([d for d in os.listdir(BBMAS_RAW_ROOT) if d.isdigit()], key=lambda x: int(x))
    swipe_feat_dir = os.path.join(BBMAS_RAW_ROOT, "Swipe_Features")
    raw_activity_rows = []
    CHUNK_KEYS = 120

    for u in user_dirs:
        ud = os.path.join(BBMAS_RAW_ROOT, u)
        if not os.path.isdir(ud):
            continue
        files = os.listdir(ud)
        kb_files = [f for f in files if "HandPhone_Keyboard" in f and f.endswith(".csv")]
        tc_files = [f for f in files if "HandPhone_TouchEvent" in f and f.endswith(".csv")]
        ac_files = [f for f in files if "HandPhone_Accelerometer" in f and f.endswith(".csv")]
        if not kb_files or not tc_files:
            continue

        acc_dyn = None
        if ac_files:
            try:
                adf = pd.read_csv(os.path.join(ud, ac_files[0]), usecols=["Xvalue", "Yvalue", "Zvalue"])
                if len(adf) > 100:
                    mag = np.sqrt(adf["Xvalue"]**2 + adf["Yvalue"]**2 + adf["Zvalue"]**2)
                    acc_dyn = np.abs(mag - mag.median()).values
            except Exception:
                acc_dyn = None

        kdf = pd.read_csv(os.path.join(ud, kb_files[0]))
        presses = kdf[kdf["direction"] == 0].copy()
        if len(presses) < 30:
            continue
        presses["ts"] = pd.to_datetime(presses["time"], errors="coerce")
        presses = presses.dropna(subset=["ts"]).reset_index(drop=True)
        presses["dt"] = presses["ts"].diff().dt.total_seconds()

        sf_path = os.path.join(swipe_feat_dir, f"{u}_Phone.csv")
        strokes_df = None
        if os.path.exists(sf_path):
            try:
                sdf = pd.read_csv(sf_path)
                if len(sdf) > 0:
                    strokes_df = pd.DataFrame({
                        "duration_ms": sdf["tottime"].clip(16.0, 2656.0),
                        "dist_px": sdf["eucliddist"].clip(7.5, 1450.0),
                        "speed_pxps": (sdf["vmean"] * 1000.0).clip(60.0, 14000.0),
                        "contact_area": np.where(
                            sdf["areamean"] > 1.0,
                            (sdf["areamean"] / 192.0).clip(0.0175, 0.0363),
                            (sdf["areamean"] * 0.05).clip(0.0175, 0.0363)
                        ),
                        "x_mean": ((sdf["minx"] + sdf["maxx"]) / 2.0).clip(45.0, 1440.0),
                        "y_mean": ((sdf["miny"] + sdf["maxy"]) / 2.0).clip(15.0, 2560.0),
                    })
            except Exception:
                strokes_df = None

        if strokes_df is None or len(strokes_df) == 0:
            continue

        n_chunks = max(1, len(presses) // CHUNK_KEYS)
        strokes_per_chunk = max(1, len(strokes_df) / n_chunks)

        for c_idx in range(n_chunks):
            kb_chunk = presses.iloc[c_idx * CHUNK_KEYS : (c_idx + 1) * CHUNK_KEYS]
            if len(kb_chunk) < 25:
                continue
            valid_dt = kb_chunk["dt"][(kb_chunk["dt"] > 0.02) & (kb_chunk["dt"] < 3.0)]
            typing_spd = float(np.clip(60.0 / valid_dt.mean(), 55.0, 400.0)) if len(valid_dt) > 0 else 180.0
            err_rate   = float((kb_chunk["key"] == "BACKSPACE").mean())

            s_start = int(round(c_idx * strokes_per_chunk))
            s_end   = int(round((c_idx + 1) * strokes_per_chunk))
            st_chunk = strokes_df.iloc[s_start : max(s_start + 1, s_end)]
            if len(st_chunk) == 0:
                st_chunk = strokes_df

            if acc_dyn is not None and len(acc_dyn) >= n_chunks:
                a_start = int(round(c_idx * (len(acc_dyn) / n_chunks)))
                a_end   = int(round((c_idx + 1) * (len(acc_dyn) / n_chunks)))
                win_acc = float(np.mean(acc_dyn[a_start : max(a_start + 1, a_end)]))
                ctx_walk = 1 if win_acc > 0.25 else 0
            else:
                ctx_walk = 0

            act_id = f"{u}_{c_idx+1:03d}"
            raw_activity_rows.append({
                "user_id": int(u),
                "activity_id": act_id,
                "task_type": "writing",
                "context": "walking" if ctx_walk == 1 else "sitting",
                "touch_duration_mean_ms": round(float(st_chunk["duration_ms"].mean()), 2),
                "swipe_speed_mean_pxps": round(float(st_chunk["speed_pxps"].mean()), 2),
                "swipe_distance_mean_px": round(float(st_chunk["dist_px"].mean()), 2),
                "typing_speed_kpm": round(typing_spd, 2),
                "backspace_rate": round(err_rate, 4),
                "contact_size_mean": round(float(st_chunk["contact_area"].mean()), 4),
                "touch_x_mean": round(float(st_chunk["x_mean"].mean()), 2),
                "touch_y_mean": round(float(st_chunk["y_mean"].mean()), 2),
            })
    raw_bbmas_df = pd.DataFrame(raw_activity_rows)
    raw_bbmas_df.to_csv(RAW_BBMAS_CSV, index=False)

print(f"      BB-MAS loaded: {raw_bbmas_df.shape[0]} rows, {raw_bbmas_df['user_id'].nunique()} users")

# 2. Apply Identical Multi-Domain Biomechanical Fitts' Law Formula
print("\n[2/2] Computing Multi-Feature Biomechanical Metrics & Balanced Target Labels...")
ready_test_rows = []

for _, row in raw_bbmas_df.iterrows():
    walk    = 1 if str(row['context']).lower() == 'walking' else 0
    dur_ms  = float(np.clip(row['touch_duration_mean_ms'] if pd.notna(row['touch_duration_mean_ms']) else 350.0, 16.0, 2656.0))
    sw_spd  = float(np.clip(row['swipe_speed_mean_pxps'] if pd.notna(row['swipe_speed_mean_pxps']) else 850.0, 60.0, 14000.0))
    sw_dst  = float(np.clip(row['swipe_distance_mean_px'] if pd.notna(row['swipe_distance_mean_px']) else 300.0, 7.5, 1450.0))
    typ_spd = float(np.clip(row['typing_speed_kpm'] if pd.notna(row['typing_speed_kpm']) else 185.0, 50.0, 400.0))
    bk_rate = float(np.clip(row['backspace_rate'] if pd.notna(row['backspace_rate']) else 0.085, 0.0, 0.50))
    c_size  = float(np.clip(row['contact_size_mean'] if pd.notna(row['contact_size_mean']) else 0.025, 0.0175, 0.0363))
    tx      = float(np.clip(row['touch_x_mean'] if pd.notna(row['touch_x_mean']) else 600.0, 45.0, 1605.0))
    ty      = float(np.clip(row['touch_y_mean'] if pd.notna(row['touch_y_mean']) else 750.0, 12.0, 1925.0))

    # Normalized physical factors in [0, 1]
    dur_norm     = float(np.clip((dur_ms - 120.0) / (750.0 - 120.0), 0.0, 1.0))
    contact_norm = float(np.clip((c_size - 0.019) / (0.032 - 0.019), 0.0, 1.0))
    swipe_norm   = float(np.clip((sw_spd - 500.0) / (2200.0 - 500.0), 0.0, 1.0))
    reach_norm   = float(np.clip(((tx - 650.0) / 550.0)**2 * 0.45 + ((ty - 750.0) / 650.0)**2 * 0.55, 0.0, 1.0))
    err_norm     = float(np.clip(bk_rate / 0.20, 0.0, 1.0))

    # A. Physical Touch Target Miss Rate (Distinct from backspace_rate!)
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

    # B. Multi-Factor Cognitive-Motor Reaction Latency (ms)
    reaction_ms = float(np.clip(
        360.0 - 0.70 * typ_spd + 0.18 * dur_ms + 42.0 * walk + 0.06 * sw_dst + 45.0 * err_norm,
        180.0, 680.0
    ))
    reaction_norm = float(np.clip((reaction_ms - 180.0) / (620.0 - 180.0), 0.0, 1.0))

    # C. Multi-Domain Biomechanical Decision Score (Connects ALL 13 features!)
    p_walk        = 1.0 if walk == 1 else 0.0
    p_touch       = (1.0 if dur_ms > 460.0 else 0.0) + (0.8 if c_size > 0.0254 else 0.0)
    p_swipe_reach = 0.7 if (sw_dst > 365.0 or ty > 920.0 or sw_spd > 1480.0 or tx > 900.0) else 0.0
    p_typing      = (1.0 if bk_rate > 0.095 else 0.0) + (0.6 if typ_spd < 174.0 else 0.0)
    p_error       = (1.2 if miss_rate > 0.165 else 0.0) + (0.8 if reaction_ms > 348.0 else 0.0)

    total_score = p_walk + p_touch + p_swipe_reach + p_typing + p_error

    if total_score >= 4.75:
        target_btn = 3  # Extra Large (64dp)
    elif total_score >= 3.25:
        target_btn = 2  # Large (56dp)
    elif total_score >= 1.75:
        target_btn = 1  # Default (48dp)
    else:
        target_btn = 0  # Small (40dp)

    ready_test_rows.append({
        "user_id": f"bbmas_{row['user_id']}",
        "session_id": f"s_{row['activity_id']}",
        "source_dataset": "BB-MAS",
        "context_is_walking": walk,
        "touch_duration_mean_ms": round(dur_ms, 2),
        "swipe_speed_mean_pxps": round(sw_spd, 2),
        "swipe_distance_mean_px": round(sw_dst, 2),
        "typing_speed_kpm": round(typ_spd, 2),
        "backspace_rate": round(bk_rate, 4),
        "contact_size_mean": round(c_size, 4),
        "touch_x_mean": round(tx, 2),
        "touch_y_mean": round(ty, 2),
        "tap_accuracy": round(tap_acc, 4),
        "miss_rate": round(miss_rate, 4),
        "reaction_time_mean_ms": round(reaction_ms, 2),
        "reaction_time_norm": round(reaction_norm, 4),
        "target_button_size": target_btn
    })

ready_bbmas_df = pd.DataFrame(ready_test_rows)
ready_out_csv  = os.path.join(OUT_DIR, "bbmas_test_ready_dataset.csv")
ready_bbmas_df.to_csv(ready_out_csv, index=False)

print(f"\n[OK] Saved BB-MAS Ready Test Dataset: {ready_out_csv}")
print(f"     Shape: {ready_bbmas_df.shape[0]} rows x {ready_bbmas_df.shape[1]} columns | Unique Users: {ready_bbmas_df['user_id'].nunique()}")
print(f"     Null values across all 17 columns: {ready_bbmas_df.isna().sum().sum()}")
print(f"     Class Distribution (target_button_size):")
print("    ", ready_bbmas_df["target_button_size"].value_counts().sort_index().to_dict())
print("     Percentages (%):", (ready_bbmas_df['target_button_size'].value_counts(normalize=True).sort_index() * 100).round(2).to_dict())
