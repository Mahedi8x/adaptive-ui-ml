import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure output directories exist
BASE_DIR = r"C:\Users\Mehedi\adavtive_model\adavtive_dataset"
DATA_PATH = os.path.join(BASE_DIR, "processed", "final_ready_dataset.csv")
FIG_DIR = os.path.join(BASE_DIR, "paper_figures", "eda")
os.makedirs(FIG_DIR, exist_ok=True)

print("="*60)
print("  Running Deep Exploratory Data Analysis (EDA) Pipeline")
print("="*60)

# 1. Load Data
df = pd.read_csv(DATA_PATH)
print(f"Loaded Dataset: {df.shape[0]} rows, {df.shape[1]} columns, {df['user_id'].nunique()} unique users.")

# 2. Comprehensive Statistical Summary Table (Mean, Std, Median, IQR, Skewness, Kurtosis)
features_to_analyze = [
    'touch_duration_mean_ms', 'swipe_speed_mean_pxps', 'swipe_distance_mean_px',
    'contact_size_mean', 'tap_accuracy', 'miss_rate', 'reaction_time_mean_ms'
]

summary_stats = []
for col in features_to_analyze:
    vals = df[col].dropna()
    q25, q75 = vals.quantile(0.25), vals.quantile(0.75)
    summary_stats.append({
        'Feature': col,
        'Mean': round(vals.mean(), 2),
        'Std Dev': round(vals.std(), 2),
        'Median': round(vals.median(), 2),
        'Q25': round(q25, 2),
        'Q75': round(q75, 2),
        'IQR': round(q75 - q25, 2),
        'Min': round(vals.min(), 2),
        'Max': round(vals.max(), 2),
        'Skewness': round(float(stats.skew(vals)), 2),
        'Kurtosis': round(float(stats.kurtosis(vals)), 2)
    })

stats_df = pd.DataFrame(summary_stats)
stats_csv = os.path.join(FIG_DIR, "eda_summary_statistics_table.csv")
stats_df.to_csv(stats_csv, index=False)
print(f"\n[Table Generated] Statistical Summary saved to: {stats_csv}")
print(stats_df.to_string(index=False))

# 3. Figure 1: Biomechanical Feature Distributions (Histograms & KDE)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
plot_feats = [
    ('touch_duration_mean_ms', 'Touch Duration (ms)', 0, 0),
    ('swipe_speed_mean_pxps', 'Swipe Velocity (px/s)', 0, 1),
    ('swipe_distance_mean_px', 'Swipe Distance (px)', 0, 2),
    ('miss_rate', 'Miss Tap Rate (0.0 - 1.0)', 1, 0),
    ('reaction_time_mean_ms', 'Reaction Time (ms)', 1, 1),
    ('contact_size_mean', 'Normalized Contact Size', 1, 2),
]

for idx, (col, label, r, c) in enumerate(plot_feats):
    ax = axes[r, c]
    sns.histplot(df[col], kde=True, ax=ax, color=palette[idx], bins=35, edgecolor='black', alpha=0.6)
    ax.set_title(f'Distribution of {label}', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel('Sample Density', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig1_path = os.path.join(FIG_DIR, "fig1_feature_distributions.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"\n[Fig 1 Saved] Feature Distributions: {fig1_path}")

# 4. Figure 2: Biomechanical Outliers Boxplots (Before vs Physiologically Validated Range)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
box_cols = [
    ('touch_duration_mean_ms', 'Touch Duration (ms)', '#3498db'),
    ('swipe_speed_mean_pxps', 'Swipe Velocity (px/s)', '#e74c3c'),
    ('reaction_time_mean_ms', 'Reaction Time (ms)', '#2ecc71')
]

for idx, (col, label, color) in enumerate(box_cols):
    ax = axes[idx]
    sns.boxplot(y=df[col], ax=ax, color=color, width=0.35, fliersize=3)
    ax.set_title(f'Outlier Dispersion: {label}', fontsize=12, fontweight='bold')
    ax.set_ylabel(label, fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig2_path = os.path.join(FIG_DIR, "fig2_outlier_boxplots.png")
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"[Fig 2 Saved] Outlier Boxplots: {fig2_path}")

# 5. Figure 3: Physical Context Impact (Sitting vs Walking) with Statistical Hypothesis Testing
sitting_miss = df[df['context_is_walking'] == 0]['miss_rate']
walking_miss = df[df['context_is_walking'] == 1]['miss_rate']
t_stat_miss, p_val_miss = stats.ttest_ind(walking_miss, sitting_miss, equal_var=False)

sitting_rt = df[df['context_is_walking'] == 0]['reaction_time_mean_ms']
walking_rt = df[df['context_is_walking'] == 1]['reaction_time_mean_ms']
t_stat_rt, p_val_rt = stats.ttest_ind(walking_rt, sitting_rt, equal_var=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot Miss Rate by Context
sns.violinplot(x='context_is_walking', y='miss_rate', data=df, ax=axes[0], palette=['#2b5c8f', '#e63946'], inner='quartile')
axes[0].set_xticklabels(['Stationary / Sitting (0)', 'Locomotive / Walking (1)'], fontsize=11)
axes[0].set_title(f'Miss Tap Rate by Context\n(Welch t-test: t={t_stat_miss:.2f}, p={p_val_miss:.2e})', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Miss Tap Rate', fontsize=11)
axes[0].grid(True, linestyle=':', alpha=0.6)

# Plot Reaction Time by Context
sns.violinplot(x='context_is_walking', y='reaction_time_mean_ms', data=df, ax=axes[1], palette=['#2b5c8f', '#e63946'], inner='quartile')
axes[1].set_xticklabels(['Stationary / Sitting (0)', 'Locomotive / Walking (1)'], fontsize=11)
axes[1].set_title(f'Reaction Latency by Context\n(Welch t-test: t={t_stat_rt:.2f}, p={p_val_rt:.2e})', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Reaction Time (ms)', fontsize=11)
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig3_path = os.path.join(FIG_DIR, "fig3_context_impact_sitting_vs_walking.png")
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"[Fig 3 Saved] Context Impact (Sitting vs Walking): {fig3_path}")

# 6. Figure 4: Target UI Class Distribution & Imbalance Ratio
class_counts = df['target_button_size'].value_counts().sort_index()
labels = ['Small (40dp)', 'Default (48dp)', 'Large (56dp)', 'Extra Large (64dp)']

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, class_counts, color=['#3a86ff', '#8338ec', '#ff006e', '#fb5607'], edgecolor='black', width=0.55)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 70, f"{yval:,} ({yval/len(df)*100:.1f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Optimal Target UI Button Size Distribution (Target Variable)', fontsize=13, fontweight='bold', pad=15)
plt.ylabel('Number of Instances', fontsize=11)
plt.ylim(0, max(class_counts) * 1.15)
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
fig4_path = os.path.join(FIG_DIR, "fig4_class_distribution_balance.png")
plt.savefig(fig4_path, dpi=300)
plt.close()
print(f"[Fig 4 Saved] Target Class Balance: {fig4_path}")

# 7. Figure 5: Complete Biomechanical Feature Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_cols = [
    'touch_duration_mean_ms', 'swipe_speed_mean_pxps', 'swipe_distance_mean_px',
    'contact_size_mean', 'tap_accuracy', 'miss_rate', 'reaction_time_mean_ms',
    'context_is_walking', 'target_button_size'
]
corr_matrix = df[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title('Biomechanical Correlation Matrix (Pearson Correlation)', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
fig5_path = os.path.join(FIG_DIR, "fig5_feature_correlation_matrix.png")
plt.savefig(fig5_path, dpi=300)
plt.close()
print(f"[Fig 5 Saved] Feature Correlation Matrix: {fig5_path}")

print("\nAll EDA figures and statistical tables generated successfully!")
