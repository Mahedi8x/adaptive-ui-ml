import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import numpy as np
import pandas as pd

# Paths
BBMAS_RAW_ROOT = r"c:\Users\Mehedi\OneDrive\Attachments\Desktop\reseach_data\BB-MAS_Dataset\BB-MAS_Dataset"
BASE_DIR       = r"C:\adavtive_dataset"
RAW_BBMAS_DIR  = os.path.join(BASE_DIR, "raw_data", "bbmas")
OUT_DIR        = os.path.join(BASE_DIR, "processed")

os.makedirs(RAW_BBMAS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

print("=================================================================")
print("  Extracting BB-MAS Raw Data & Generating Test Ready Dataset")
print("  (Using 100% Identical HMOG Formula from generate_ready_dataset.py)")
print("=================================================================")

def parse_bbmas_time(s):
    if not isinstance(s, str) or s == "null" or not s.strip():
        return pd.NaT
    if s.count(":") == 3:
        idx = s.rfind(":")
        s = s[:idx] + "." + s[idx+1:]
    return pd.to_datetime(s, errors="coerce")

user_dirs = sorted(
    [d for d in os.listdir(BBMAS_RAW_ROOT) if d.isdigit()],
    key=lambda x: int(x)
)
swipe_feat_dir = os.path.join(BBMAS_RAW_ROOT, "Swipe_Features")

raw_activity_rows = []
ready_test_rows   = []

CHUNK_KEYS = 120  # ~1-1.5 min typing/interaction session window per activity

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

    # Load Accelerometer dynamic motion profile if available
    acc_dyn = None
    if ac_files:
        try:
            adf = pd.read_csv(os.path.join(ud, ac_files[0]), usecols=["Xvalue", "Yvalue", "Zvalue"])
            if len(adf) > 100:
                mag = np.sqrt(adf["Xvalue"]**2 + adf["Yvalue"]**2 + adf["Zvalue"]**2)
                acc_dyn = np.abs(mag - mag.median()).values
        except Exception:
            acc_dyn = None

    # 1. Load Keyboard Events (direction == 0 is Key Press)
    kdf = pd.read_csv(os.path.join(ud, kb_files[0]))
    presses = kdf[kdf["direction"] == 0].copy()
    if len(presses) < 30:
        continue
    presses["ts"] = pd.to_datetime(presses["time"], errors="coerce")
    presses = presses.dropna(subset=["ts"]).reset_index(drop=True)
    presses["dt"] = presses["ts"].diff().dt.total_seconds()

    # 2. Load Swipe Features (or fallback to raw TouchEvent strokes)
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
        tdf = pd.read_csv(os.path.join(ud, tc_files[0]))
        downs = tdf[tdf["actionType"] == 0.0].reset_index(drop=True)
        ups   = tdf[tdf["actionType"] == 1.0].reset_index(drop=True)
        n_st  = min(len(downs), len(ups))
        if n_st == 0:
            continue
        t_down = downs["time"].iloc[:n_st].apply(parse_bbmas_time)
        t_up   = ups["time"].iloc[:n_st].apply(parse_bbmas_time)
        dur_ms = ((t_up - t_down).dt.total_seconds() * 1000.0).fillna(400.0).clip(16.0, 2656.0)
        dx = ups["Xvalue"].iloc[:n_st].values - downs["Xvalue"].iloc[:n_st].values
        dy = ups["Yvalue"].iloc[:n_st].values - downs["Yvalue"].iloc[:n_st].values
        dist_px = np.sqrt(dx**2 + dy**2).clip(7.5, 1450.0)
        speed_pxps = (dist_px / (dur_ms / 1000.0)).clip(60.0, 14000.0)
        tmaj = downs["touchMajor"].iloc[:n_st].fillna(4.7).values
        c_area = np.where(tmaj > 1.0, tmaj / 192.0, tmaj * 0.05).clip(0.0175, 0.0363)
        strokes_df = pd.DataFrame({
            "duration_ms": dur_ms.values,
            "dist_px": dist_px,
            "speed_pxps": speed_pxps,
            "contact_area": c_area,
            "x_mean": ((downs["Xvalue"].iloc[:n_st].values + ups["Xvalue"].iloc[:n_st].values) / 2.0).clip(45.0, 1440.0),
            "y_mean": ((downs["Yvalue"].iloc[:n_st].values + ups["Yvalue"].iloc[:n_st].values) / 2.0).clip(15.0, 2560.0),
        })

    # 3. Segment user timeline into activity sessions and pair with touch/swipe chunks
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

        t_dur_ms   = float(st_chunk["duration_ms"].mean())
        sw_spd     = float(st_chunk["speed_pxps"].mean())
        sw_dist    = float(st_chunk["dist_px"].mean())
        c_size     = float(st_chunk["contact_area"].mean())
        tx_mean    = float(st_chunk["x_mean"].mean())
        ty_mean    = float(st_chunk["y_mean"].mean())

        # Determine sitting vs walking/moving context from real HandPhone_Accelerometer window
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
            "touch_duration_mean_ms": round(t_dur_ms, 2),
            "swipe_speed_mean_pxps": round(sw_spd, 2),
            "swipe_distance_mean_px": round(sw_dist, 2),
            "typing_speed_kpm": round(typing_spd, 2),
            "backspace_rate": round(err_rate, 4),
            "contact_size_mean": round(c_size, 4),
            "touch_x_mean": round(tx_mean, 2),
            "touch_y_mean": round(ty_mean, 2),
        })

        # =====================================================================
        # EXACT HMOG FORMULA FROM generate_ready_dataset.py (Lines 35 - 73)
        # =====================================================================
        miss_rate = float(np.clip(err_rate, 0.02, 0.40))
        tap_acc   = 1.0 - miss_rate

        reaction_ms   = float(np.clip(600.0 - (typing_spd * 1.5), 180.0, 800.0))
        reaction_norm = (reaction_ms - 180.0) / (800.0 - 180.0)

        if miss_rate > 0.16 or reaction_norm > 0.65:
            target_btn = 3  # Extra Large (64dp)
        elif miss_rate > 0.11 or reaction_norm > 0.45:
            target_btn = 2  # Large (56dp)
        elif miss_rate > 0.06:
            target_btn = 1  # Medium / Default (48dp)
        else:
            target_btn = 0  # Small (40dp)

        ready_test_rows.append({
            "user_id": f"bbmas_{u}",
            "session_id": f"s_{act_id}",
            "source_dataset": "BB-MAS",
            "context_is_walking": ctx_walk,
            "touch_duration_mean_ms": round(t_dur_ms, 2),
            "swipe_speed_mean_pxps": round(sw_spd, 2),
            "swipe_distance_mean_px": round(sw_dist, 2),
            "typing_speed_kpm": round(typing_spd, 2),
            "backspace_rate": round(err_rate, 4),
            "contact_size_mean": round(c_size, 4),
            "touch_x_mean": round(tx_mean, 2),
            "touch_y_mean": round(ty_mean, 2),
            "tap_accuracy": round(tap_acc, 4),
            "miss_rate": round(miss_rate, 4),
            "reaction_time_mean_ms": round(reaction_ms, 2),
            "reaction_time_norm": round(reaction_norm, 4),
            "target_button_size": target_btn
        })

raw_bbmas_df   = pd.DataFrame(raw_activity_rows)
ready_bbmas_df = pd.DataFrame(ready_test_rows)

raw_out_csv   = os.path.join(RAW_BBMAS_DIR, "bbmas_activity_features.csv")
ready_out_csv = os.path.join(OUT_DIR, "bbmas_test_ready_dataset.csv")

raw_bbmas_df.to_csv(raw_out_csv, index=False)
ready_bbmas_df.to_csv(ready_out_csv, index=False)

print(f"\n[1] Saved BB-MAS Raw Activity Features: {raw_out_csv}")
print(f"    Shape: {raw_bbmas_df.shape[0]} rows x {raw_bbmas_df.shape[1]} columns | Unique Users: {raw_bbmas_df['user_id'].nunique()}")

print(f"\n[2] Saved BB-MAS Ready Test Dataset: {ready_out_csv}")
print(f"    Shape: {ready_bbmas_df.shape[0]} rows x {ready_bbmas_df.shape[1]} columns | Unique Users: {ready_bbmas_df['user_id'].nunique()}")
print(f"    Class Distribution (target_button_size):")
print("   ", ready_bbmas_df["target_button_size"].value_counts().sort_index().to_dict())
print(f"    Null values across all 17 columns: {ready_bbmas_df.isna().sum().sum()}")
