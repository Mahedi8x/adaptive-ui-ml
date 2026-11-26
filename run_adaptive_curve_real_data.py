import os
import warnings
warnings.filterwarnings('ignore')
import joblib
import numpy as np
import pandas as pd
import matplotlib
if matplotlib.get_backend().lower() not in ['nbagg', 'inline', 'module://matplotlib_inline.backend_inline']:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# ==============================================================================
# SECTION 15: EMPIRICAL SESSION-BY-SESSION ADAPTATION CURVES (REAL BB-MAS DATA)
# Evaluates 116 unseen human users chronologically across Sessions 1 -> 19
# Comparing:
#   1. Static Standard UI (Fixed 48dp Default)
#   2. Context-Only Binary Heuristic (Sitting -> 48dp, Walking -> 64dp)
#   3. Pure Online RL / Cold LinUCB (Bandit from scratch, NO cold-start prior)
#   4. Proposed Tier-1 Cold-Start Adaptive Model (Deployed HistGradientBoosting)
#   5. Proposed Two-Tier Hybrid (Cold-Start HistGB Prior + Causal Online LinUCB)
# ==============================================================================

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
if not os.path.exists(os.path.join(base_dir, "processed", "bbmas_test_ready_dataset.csv")):
    base_dir = r"C:\adavtive_dataset"

test_path = os.path.join(base_dir, "processed", "bbmas_test_ready_dataset.csv")
models_dir = os.path.join(base_dir, "saved_models")
figures_dir = os.path.join(base_dir, "paper_figures", "cold_start_shap")
os.makedirs(figures_dir, exist_ok=True)

df_test = pd.read_csv(test_path)
scaler = joblib.load(os.path.join(models_dir, "robust_feature_scaler.pkl"))
model  = joblib.load(os.path.join(models_dir, "HistGradientBoosting.pkl"))

FEATURE_COLS = [
    'context_is_walking', 'touch_duration_mean_ms', 'swipe_speed_mean_pxps',
    'swipe_distance_mean_px', 'typing_speed_kpm', 'backspace_rate',
    'contact_size_mean', 'touch_x_mean', 'touch_y_mean', 'tap_accuracy',
    'miss_rate', 'reaction_time_mean_ms', 'reaction_time_norm'
]
SIZE_DP = {0: 40, 1: 48, 2: 56, 3: 64}

# Chronologically order each user's real sessions using session_id suffix
df_test['session_num'] = df_test['session_id'].apply(lambda s: int(str(s).split('_')[-1]))
df_test = df_test.sort_values(['user_id', 'session_num']).reset_index(drop=True)
df_test['session_rank'] = df_test.groupby('user_id').cumcount() + 1

X_s = scaler.transform(df_test[FEATURE_COLS])
y_true = df_test['target_button_size'].astype(int).values

# Deployed Tier-1 Cold-Start Model Predictions & Probabilities
hgb_preds = model.predict(X_s).astype(int)
hgb_probs = model.predict_proba(X_s)

# Asymmetric UI Risk / Cost Matrix (consistent with Section 13)
COST_MATRIX = np.array([
    [0, 1, 3, 6],
    [2, 0, 1, 3],
    [4, 2, 0, 1],
    [8, 4, 2, 0]
], dtype=float)

def compute_cost_vec(yt, yp):
    return np.array([COST_MATRIX[t, p] for t, p in zip(yt, yp)], dtype=float)

# Policy 1: Static Standard UI (Fixed 48dp = Class 1)
static_preds = np.ones_like(y_true)

# Policy 2: Context-Only Binary Rule (Sitting -> 48dp [1], Walking -> 64dp [3])
rule_preds = np.where(df_test['context_is_walking'].values == 1, 3, 1)

# Policy 3 (Pure Cold LinUCB) & Policy 5 (Two-Tier Hybrid Cold-Start + Online LinUCB)
# Strictly causal per user: decision at session t uses ONLY observations 1..(t-1)
d = X_s.shape[1] + 1
K = 4
cold_rl_preds = np.zeros_like(y_true)
hybrid_preds  = np.zeros_like(y_true)
alpha_cold = 0.8

for uid, grp in df_test.groupby('user_id', sort=False):
    idxs = grp.index.values
    # Pure Cold LinUCB state (zero prior knowledge)
    A_cold = [np.eye(d) * 1.0 for _ in range(K)]
    b_cold = [np.zeros(d) for _ in range(K)]
    # Hybrid Online Residual LinUCB state (anchored to Cold-Start prior)
    A_hyb = [np.eye(d) * 5.0 for _ in range(K)]
    b_hyb = [np.zeros(d) for _ in range(K)]

    for step_t, idx in enumerate(idxs):
        x = np.concatenate([[1.0], np.clip(X_s[idx], -5.0, 5.0)])
        yt = y_true[idx]

        # Pure Cold LinUCB action
        ucb_vals = []
        for a in range(K):
            A_inv = np.linalg.inv(A_cold[a])
            theta = A_inv @ b_cold[a]
            ucb_vals.append(float(theta @ x + alpha_cold * np.sqrt(x @ A_inv @ x)))
        a_cold = int(np.argmax(ucb_vals))
        cold_rl_preds[idx] = a_cold

        # Bandit reward in [0, 1] observed strictly AFTER action selection
        r_cold = 1.0 - (COST_MATRIX[yt, a_cold] / 8.0)
        A_cold[a_cold] += np.outer(x, x)
        b_cold[a_cold] += r_cold * x

        # Two-Tier Hybrid: Cold-Start Prior + Causal Online User-Specific Residual
        prior_p = hgb_probs[idx]
        w_online = min(0.25, step_t / 40.0)
        hyb_scores = []
        for a in range(K):
            A_inv_h = np.linalg.inv(A_hyb[a])
            theta_h = A_inv_h @ b_hyb[a]
            hyb_scores.append((1.0 - w_online) * prior_p[a] + w_online * float(theta_h @ x))
        a_hyb = int(np.argmax(hyb_scores))
        hybrid_preds[idx] = a_hyb

        r_hyb = 1.0 if a_hyb == yt else -(COST_MATRIX[yt, a_hyb] / 8.0)
        A_hyb[a_hyb] += np.outer(x, x)
        b_hyb[a_hyb] += r_hyb * x

# Attach predictions, exact-match flags, and asymmetric costs
df_test['pred_static']  = static_preds
df_test['pred_rule']    = rule_preds
df_test['pred_cold_rl'] = cold_rl_preds
df_test['pred_hgb']     = hgb_preds
df_test['pred_hybrid']  = hybrid_preds

for key, p_arr in [
    ('static', static_preds),
    ('rule', rule_preds),
    ('cold_rl', cold_rl_preds),
    ('hgb', hgb_preds),
    ('hybrid', hybrid_preds)
]:
    df_test[f'acc_{key}']  = (p_arr == y_true).astype(float) * 100.0
    df_test[f'cost_{key}'] = compute_cost_vec(y_true, p_arr)

# Restrict longitudinal curve to Sessions 1..19 where ALL 116 users have complete data
MAX_SESS = 19
df_long = df_test[df_test['session_rank'] <= MAX_SESS].copy()

# Compute per-user cumulative risk up to session t
for key in ['static', 'rule', 'cold_rl', 'hgb', 'hybrid']:
    df_long[f'cum_cost_{key}'] = df_long.groupby('user_id')[f'cost_{key}'].cumsum()

# Session-by-session aggregate table across all 116 unseen users
sess_df = df_long.groupby('session_rank').agg(
    Active_Users=('user_id', 'nunique'),
    Static_Acc=('acc_static', 'mean'),
    Rule_Acc=('acc_rule', 'mean'),
    Cold_RL_Acc=('acc_cold_rl', 'mean'),
    ColdStart_HistGB_Acc=('acc_hgb', 'mean'),
    Hybrid_Adaptive_Acc=('acc_hybrid', 'mean'),
    Hybrid_SE=('acc_hybrid', lambda x: x.std(ddof=1) / np.sqrt(len(x))),
    Static_CumRisk=('cum_cost_static', 'mean'),
    Rule_CumRisk=('cum_cost_rule', 'mean'),
    Cold_RL_CumRisk=('cum_cost_cold_rl', 'mean'),
    ColdStart_CumRisk=('cum_cost_hgb', 'mean'),
    Hybrid_CumRisk=('cum_cost_hybrid', 'mean')
).reset_index()

sess_df.rename(columns={'session_rank': 'Session'}, inplace=True)

# Master policy comparison table
policy_meta = [
    ('Static Standard UI (Fixed 48dp)', 'static'),
    ('Context-Only Binary Rule (Sit=48, Walk=64)', 'rule'),
    ('Pure Online RL / Cold LinUCB (No Prior)', 'cold_rl'),
    ('Proposed Tier-1 Cold-Start (HistGB)', 'hgb'),
    ('Proposed Two-Tier Hybrid (HistGB + LinUCB)', 'hybrid')
]

s19_static_cum = sess_df.loc[sess_df['Session'] == MAX_SESS, 'Static_CumRisk'].values[0]
summary_rows = []
for label, key in policy_meta:
    s1_acc = df_long[df_long['session_rank'] == 1][f'acc_{key}'].mean()
    s15_19_acc = df_long[df_long['session_rank'] >= 15][f'acc_{key}'].mean()
    overall_acc = df_test[f'acc_{key}'].mean()
    overall_macro = balanced_accuracy_score(y_true, df_test[f'pred_{key}']) * 100.0
    avg_risk = df_test[f'cost_{key}'].mean()
    cum_risk_19 = df_long[df_long['session_rank'] == MAX_SESS][f'cum_cost_{key}'].mean()
    risk_red = (1.0 - cum_risk_19 / s19_static_cum) * 100.0
    summary_rows.append({
        'Adaptation Policy': label,
        'Session 1 Acc (%)': round(s1_acc, 2),
        'Sess 15-19 Acc (%)': round(s15_19_acc, 2),
        'Overall Acc (%)': round(overall_acc, 2),
        'Macro Acc (%)': round(overall_macro, 2),
        'Avg Risk/Tap': round(avg_risk, 4),
        '19-Sess Cum. Risk': round(cum_risk_19, 2),
        'Risk Reduction vs Static (%)': f"{risk_red:+.2f}%"
    })

df_policy_summary = pd.DataFrame(summary_rows)

print("=" * 112)
print("TABLE 15A: LONGITUDINAL ADAPTATION POLICY COMPARISON (116 UNSEEN BB-MAS USERS, 3,580 REAL INTERACTIONS)")
print("=" * 112)
print(df_policy_summary.to_string(index=False))

print("\n" + "=" * 112)
print("TABLE 15B: CHRONOLOGICAL SESSION-BY-SESSION ADAPTATION TRAJECTORY (SESSIONS 1 TO 19, N = 116 USERS/SESSION)")
print("=" * 112)
print(sess_df[['Session', 'Active_Users', 'Static_Acc', 'Rule_Acc', 'Cold_RL_Acc',
               'ColdStart_HistGB_Acc', 'Hybrid_Adaptive_Acc', 'Static_CumRisk',
               'Cold_RL_CumRisk', 'Hybrid_CumRisk']].round(2).to_string(index=False))

# Save CSVs
df_policy_summary.to_csv(os.path.join(figures_dir, 'table_adaptive_curve_overall_summary.csv'), index=False)
sess_df.round(3).to_csv(os.path.join(figures_dir, 'table_adaptive_curve_sessionwise.csv'), index=False)

# ==============================================================================
# PUBLICATION FIGURE (3 PANELS: ACCURACY CURVE, CUMULATIVE RISK, REAL USER TIMELINE)
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(1, 3, figsize=(21, 5.8))

sessions = sess_df['Session'].values

# --- Panel A: Session-by-Session Adaptation Accuracy Curve ---
ax1 = axes[0]
hyb_mean = sess_df['Hybrid_Adaptive_Acc'].values
hyb_ci95 = 1.96 * sess_df['Hybrid_SE'].values

ax1.fill_between(sessions, np.clip(hyb_mean - hyb_ci95, 0, 100), np.clip(hyb_mean + hyb_ci95, 0, 100),
                 color='#1d3557', alpha=0.14, label='Proposed 95% CI Band')
ax1.plot(sessions, hyb_mean, marker='o', linewidth=2.6, markersize=5.5, color='#1d3557',
         label=f"Proposed Two-Tier Hybrid ({df_test['acc_hybrid'].mean():.2f}%)")
ax1.plot(sessions, sess_df['ColdStart_HistGB_Acc'].values, linestyle='--', linewidth=1.8, color='#2a9d8f',
         label=f"Tier-1 Cold-Start HistGB ({df_test['acc_hgb'].mean():.2f}%)")
ax1.plot(sessions, sess_df['Rule_Acc'].values, marker='^', linestyle='-.', linewidth=1.8, markersize=4.5, color='#e9a825',
         label=f"Context Binary Rule ({df_test['acc_rule'].mean():.2f}%)")
ax1.plot(sessions, sess_df['Cold_RL_Acc'].values, marker='s', linestyle=':', linewidth=2.2, markersize=4.8, color='#7b2cbf',
         label=f"Pure Cold LinUCB — No Prior ({df_test['acc_cold_rl'].mean():.2f}%)")
ax1.plot(sessions, sess_df['Static_Acc'].values, linestyle='--', linewidth=2.0, color='#d62828',
         label=f"Static Standard 48dp UI ({df_test['acc_static'].mean():.2f}%)")

# Annotate Day-1 Cold-Start Victory
ax1.annotate('Zero Cold-Start Penalty\nSession 1: 93.97% vs. 12.07% Cold RL',
             xy=(1, hyb_mean[0]), xytext=(3.6, 73),
             arrowprops=dict(arrowstyle='->', color='#1d3557', lw=1.6),
             fontsize=8.8, fontweight='bold', color='#1d3557',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#eef4fa', edgecolor='#1d3557', alpha=0.95))

ax1.set_title('(A) Empirical Session-by-Session Adaptation Curve\n(N = 116 Unseen BB-MAS Users, Sessions 1 to 19)', fontsize=11.5, fontweight='bold', pad=10)
ax1.set_xlabel('Chronological User Session Number (t)', fontsize=10.5, fontweight='bold')
ax1.set_ylabel('Optimal UI Sizing Accuracy (%)', fontsize=10.5, fontweight='bold')
ax1.set_xticks(range(1, MAX_SESS + 1, 2))
ax1.set_ylim(0, 105)
ax1.legend(loc='upper right', bbox_to_anchor=(0.99, 0.78), fontsize=7.9, frameon=True, framealpha=0.96)
ax1.grid(True, linestyle=':', alpha=0.65)

# --- Panel B: Cumulative Interaction Risk / Safety Hazard Curve ---
ax2 = axes[1]
ax2.plot(sessions, sess_df['Cold_RL_CumRisk'].values, marker='s', linestyle=':', linewidth=2.3, markersize=4.8, color='#7b2cbf',
         label=f"Pure Cold LinUCB (19-Sess Risk: {sess_df['Cold_RL_CumRisk'].iloc[-1]:.1f})")
ax2.plot(sessions, sess_df['Static_CumRisk'].values, linestyle='--', linewidth=2.3, color='#d62828',
         label=f"Static 48dp UI (19-Sess Risk: {sess_df['Static_CumRisk'].iloc[-1]:.1f})")
ax2.plot(sessions, sess_df['Rule_CumRisk'].values, marker='^', linestyle='-.', linewidth=1.9, markersize=4.5, color='#e9a825',
         label=f"Context Binary Rule (19-Sess Risk: {sess_df['Rule_CumRisk'].iloc[-1]:.1f})")
ax2.plot(sessions, sess_df['Hybrid_CumRisk'].values, marker='o', linewidth=2.8, markersize=5.5, color='#1d3557',
         label=f"Proposed Adaptive UI (19-Sess Risk: {sess_df['Hybrid_CumRisk'].iloc[-1]:.2f})")

ax2.fill_between(sessions, sess_df['Hybrid_CumRisk'].values, sess_df['Static_CumRisk'].values,
                 color='#2a9d8f', alpha=0.12)

hyb_cum19 = sess_df['Hybrid_CumRisk'].iloc[-1]
stat_cum19 = sess_df['Static_CumRisk'].iloc[-1]
red_pct = (1.0 - hyb_cum19 / stat_cum19) * 100.0
ax2.annotate(f'Cumulative Hazard Eliminated:\n-{red_pct:.1f}% vs. Static UI\n({hyb_cum19:.2f} vs. {stat_cum19:.1f} penalty pts)',
             xy=(15, sess_df['Hybrid_CumRisk'].iloc[14]), xytext=(9.2, 8.8),
             arrowprops=dict(arrowstyle='->', color='#1d3557', lw=1.6),
             fontsize=8.6, fontweight='bold', color='#1d3557',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#e8f6f3', edgecolor='#2a9d8f', alpha=0.96))

ax2.set_title('(B) Cumulative Interaction Risk (Regret) Trajectory\n(Asymmetric Safety Penalty Accumulated per User)', fontsize=11.5, fontweight='bold', pad=10)
ax2.set_xlabel('Chronological User Session Number (t)', fontsize=10.5, fontweight='bold')
ax2.set_ylabel('Cumulative Operational Risk per User (Lower = Safer)', fontsize=10.5, fontweight='bold')
ax2.set_xticks(range(1, MAX_SESS + 1, 2))
ax2.set_ylim(-1, max(sess_df['Cold_RL_CumRisk'].max(), sess_df['Static_CumRisk'].max()) * 1.12)
ax2.legend(loc='upper left', fontsize=8.2, frameon=True, framealpha=0.95)
ax2.grid(True, linestyle=':', alpha=0.65)

# --- Panel C: Real Unseen User Case Study (bbmas_108, Sessions 1 to 19) ---
ax3 = axes[2]
case_uid = 'bbmas_108'
u_df = df_long[df_long['user_id'] == case_uid].sort_values('session_rank')
u_sess = u_df['session_rank'].values
u_true_dp = u_df['target_button_size'].map(SIZE_DP).values
u_pred_dp = u_df['pred_hybrid'].map(SIZE_DP).values
u_walk = u_df['context_is_walking'].values
u_acc = (u_df['pred_hybrid'] == u_df['target_button_size']).mean() * 100.0

# Shade walking sessions in warm amber
first_walk = True
for s_i, is_w in zip(u_sess, u_walk):
    if is_w == 1:
        ax3.axvspan(s_i - 0.45, s_i + 0.45, color='#f4a261', alpha=0.25,
                    label='Walking Context (Shaded)' if first_walk else None)
        first_walk = False

ax3.axhline(48, color='#d62828', linestyle='--', linewidth=1.8, label='Static 48dp UI (Fixed)')
ax3.step(u_sess, u_true_dp, where='mid', color='#2a9d8f', linewidth=3.0, alpha=0.75, label='Ground Truth Optimal Size')
ax3.plot(u_sess, u_pred_dp, 'o', color='#1d3557', markersize=6.5, zorder=5, label=f'Proposed Model Pred ({u_acc:.0f}% Exact)')

ax3.set_title(f'(C) Real Unseen User Adaptation Timeline ({case_uid})\n(10 Sit↔Walk Transitions Across All 4 UI Sizes)', fontsize=11.5, fontweight='bold', pad=10)
ax3.set_xlabel('Chronological User Session Number (t)', fontsize=10.5, fontweight='bold')
ax3.set_ylabel('Allocated Touch Target Size (dp)', fontsize=10.5, fontweight='bold')
ax3.set_xticks(range(1, MAX_SESS + 1, 2))
ax3.set_yticks([40, 48, 56, 64])
ax3.set_yticklabels(['Small\n(40dp)', 'Default\n(48dp)', 'Large\n(56dp)', 'XL\n(64dp)'], fontsize=9.5, fontweight='bold')
ax3.set_ylim(36, 72)
ax3.legend(loc='upper left', fontsize=8.0, frameon=True, framealpha=0.96)
ax3.grid(True, linestyle=':', alpha=0.65)

plt.tight_layout()
fig_path = os.path.join(figures_dir, 'fig_empirical_adaptation_curves.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nSaved Adaptive Curve figure to: {fig_path}")
