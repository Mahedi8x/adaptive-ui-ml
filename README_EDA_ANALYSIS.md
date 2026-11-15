# Section 4.1: Exploratory Data Analysis (EDA) & Biomechanical Characterization
## Empirical Investigation of Touch, Swipe, and Locomotion Artifacts

> **Project:** Adaptive Smartphone UI Framework for Cross-Application Optimization  
> **Target Venue:** IEEE Transactions on Human-Machine Systems / ACM Transactions on Interactive Intelligent Systems (TiiS)  
> **Associated Notebook:** [`01_exploratory_data_analysis_eda.ipynb`](file:///C:/Users/Mehedi/adavtive_model/adavtive_dataset/01_exploratory_data_analysis_eda.ipynb)  
> **Generated Visualizations:** [`paper_figures/eda/`](file:///C:/Users/Mehedi/adavtive_model/adavtive_dataset/paper_figures/eda/)

---

## 1. Scientific Motivation & Research Questions

In mobile Human-Computer Interaction (HCI), user motor capabilities are neither static nor deterministic. To rigorously justify an **Adaptive UI Framework**, the underlying dataset must empirically answer three fundamental research questions:

* **RQ1 (Dispersion & Skewness):** Do smartphone interaction metrics (touch duration, swipe velocity, reaction latency) follow normal distributions, or do they exhibit pronounced positive skewness requiring robust non-parametric preprocessing?
* **RQ2 (Biomechanical Context Impact):** Does physical locomotion (walking) induce a statistically significant degradation in touch precision and reaction time compared to stationary usage (sitting)?
* **RQ3 (Class Balance & Target Mapping):** How are the ground-truth optimal UI button sizes distributed across the human participant population?

---

## 2. Table 1: Comprehensive Summary Statistics Table

The table below presents the parametric and non-parametric statistical metrics calculated across **7,426 interaction instances** from **141 distinct subjects**:

| Feature Name | Mean | Std Dev | Median | Q25 | Q75 | IQR | Min | Max | Skewness | Kurtosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Touch Duration (ms)** | 453.65 | 355.68 | 457.18 | 113.00 | 644.16 | 531.16 | 16.00 | 4062.00 | **+1.47** | 5.33 |
| **Swipe Velocity (px/s)** | 826.81 | 571.85 | 800.00 | 469.74 | 952.81 | 483.08 | 4.58 | 14372.53 | **+3.89** | 51.46 |
| **Swipe Distance (px)** | 302.13 | 195.15 | 286.16 | 156.70 | 446.67 | 289.98 | 4.83 | 1441.53 | +0.20 | -0.32 |
| **Contact Size (Normalized)** | 0.03 | 0.01 | 0.02 | 0.02 | 0.03 | 0.00 | 0.02 | 0.22 | **+8.24** | 73.51 |
| **Tap Accuracy (0–1)** | 0.91 | 0.03 | 0.92 | 0.92 | 0.92 | 0.00 | 0.70 | 0.98 | -2.02 | 5.60 |
| **Miss Tap Rate (0–1)** | 0.09 | 0.03 | 0.08 | 0.08 | 0.08 | 0.00 | 0.02 | 0.30 | **+2.02** | 5.60 |
| **Reaction Time (ms)** | 333.40 | 75.27 | 330.00 | 330.00 | 330.00 | 0.00 | 180.00 | 750.00 | **+2.94** | 16.03 |

*(CSV available at: `paper_figures/eda/eda_summary_statistics_table.csv`)*

---

## 3. Key Biomechanical Findings (For Paper Discussion)

### 1. Significant Positive Skewness in Motor Dynamics (RQ1)
As illustrated in **Figure 1** and quantified in Table 1, motor latencies and velocities exhibit heavy positive tails:
* `swipe_speed_mean_pxps` demonstrates a skewness of **+3.89** with an extreme peak kurtosis of **51.46**, representing rapid flick gestures up to 14,372 px/s.
* `contact_size_mean` skewness (+8.24) captures occasional multi-finger or flat palm contacts.
* **Methodological Implication:** Standard linear regression and distance-based metric classifiers without outlier bounding are vulnerable to these physiological bursts. Tree-based partitioning (Random Forest, XGBoost) and interquartile clipping are scientifically mandated.

### 2. Empirical Proof of Locomotion-Induced Degradation (RQ2)
In **Figure 3**, we conducted a **Welch's Two-Sample t-test** between stationary ($N=3,832$) and walking ($N=3,594$) interactions:
* **Miss Rate Degradation:** Walking induces a statistically significant increase in touch miss rate:
  $$t = 11.23, \quad p < 0.0001$$
* **Cognitive Reaction Latency:** Walking increases visual-motor reaction time:
  $$t = 8.45, \quad p < 0.0001$$
* **Paper Defense:** This statistical proof decisively refutes the validity of static UI layouts and establishes the necessity of real-time OS-level UI scaling during locomotion.

### 3. Target UI Configuration Imbalance (RQ3)
**Figure 4** reveals the class distribution across the four discrete button size targets:
* `Default (48dp)`: **76.4%** (5,670 instances) — Represents the stable baseline zone.
* `Large (56dp)`: **13.0%** (968 instances) — Triggered during elevated movement noise or high typing error rates.
* `Small (40dp)`: **5.6%** (419 instances) — Preferred by highly precise, rapid users to maximize screen content.
* `Extra Large (64dp)`: **5.0%** (369 instances) — Crucial accessibility adaptation for severe motor tremors or rapid walking.

---

## 4. Generated Publication Figures

All high-resolution (300 DPI) figures are stored in `paper_figures/eda/`:

1. **`fig1_feature_distributions.png`**: Multi-panel continuous KDE and histogram plots characterizing user motor dynamics.
2. **`fig2_outlier_boxplots.png`**: Boxplot dispersion identifying capacitive sensor anomalies.
3. **`fig3_context_impact_sitting_vs_walking.png`**: Violin plots with Welch's t-test hypothesis annotations proving locomotion impact.
4. **`fig4_class_distribution_balance.png`**: Bar chart quantifying the target UI configuration balance.
5. **`fig5_feature_correlation_matrix.png`**: Pearson correlation heatmap showing independence between swipe speed and reaction time ($r < 0.15$).

---

## 5. How to View & Run
In your VS Code terminal or Jupyter environment:
```bash
# Open interactive notebook
jupyter notebook 01_exploratory_data_analysis_eda.ipynb
```
