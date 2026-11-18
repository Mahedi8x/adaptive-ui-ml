# Section 4.1: Exploratory Data Analysis (EDA) & Biomechanical Characterization
## Empirical Investigation of Train (HMOG) and External Test (BB-MAS) Datasets

> **Project:** Adaptive Smartphone UI Framework for Cross-Application Optimization  
> **Target Venue:** IEEE Transactions on Human-Machine Systems / ACM Transactions on Interactive Intelligent Systems (TiiS)  
> **Associated Notebook:** [`01_exploratory_data_analysis_eda.ipynb`](file:///C:/adavtive_dataset/01_exploratory_data_analysis_eda.ipynb)  
> **Generated Visualizations:** [`paper_figures/eda/`](file:///C:/adavtive_dataset/paper_figures/eda/)

---

## 1. Dataset Overview & Unified Formula

Following strict cross-dataset validation protocol, our study evaluates two independent real-user smartphone datasets processed under **100% identical feature and Fitts' Law target equations** (`0` null values across all 17 columns):

* **Training Dataset (`HMOG`):** **7,194 interaction sessions** from **100 unique subjects** ([`final_ready_dataset.csv`](file:///C:/adavtive_dataset/processed/final_ready_dataset.csv))
* **External Testing Dataset (`SU-AIS BB-MAS`):** **4,514 interaction sessions** from **116 unique unseen subjects** ([`bbmas_test_ready_dataset.csv`](file:///C:/adavtive_dataset/processed/bbmas_test_ready_dataset.csv))
* **Total Combined Benchmark:** **11,708 sessions** across **216 distinct human participants**.

---

## 2. Table 1: Side-by-Side Statistical Summary (Train vs. External Test)

| Dataset Split | Feature Name | Mean | Std Dev | Median | Q25 | Q75 | IQR | Min | Max | Skewness | Kurtosis |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train (HMOG)** | `touch_duration_mean_ms` | 445.21 | 336.13 | 455.98 | 112.00 | 638.85 | 526.85 | 16.00 | 2656.20 | +1.11 | +2.48 |
| **Test (BB-MAS)** | `touch_duration_mean_ms` | 476.37 | 372.96 | 359.59 | 252.00 | 551.88 | 299.88 | 50.67 | 2656.00 | +2.65 | +9.08 |
| **Train (HMOG)** | `swipe_speed_mean_pxps` | 851.49 | 563.83 | 800.00 | 497.39 | 967.32 | 469.93 | 61.48 | 14372.53 | +4.14 | +55.64 |
| **Test (BB-MAS)** | `swipe_speed_mean_pxps` | 1538.65 | 1052.98 | 1265.63 | 774.68 | 2051.46 | 1276.78 | 60.00 | 8449.38 | +1.36 | +2.47 |
| **Train (HMOG)** | `swipe_distance_mean_px` | 311.36 | 191.26 | 302.00 | 213.31 | 451.56 | 238.25 | 7.47 | 1441.53 | +0.19 | -0.22 |
| **Test (BB-MAS)** | `swipe_distance_mean_px` | 392.11 | 194.45 | 357.72 | 257.30 | 497.75 | 240.45 | 7.50 | 1359.79 | +0.93 | +1.27 |
| **Train (HMOG)** | `typing_speed_kpm` | 183.90 | 34.16 | 180.00 | 180.00 | 180.00 | 0.00 | 55.10 | 400.00 | +1.26 | +5.16 |
| **Test (BB-MAS)** | `typing_speed_kpm` | 195.41 | 39.06 | 191.23 | 167.97 | 219.67 | 51.69 | 99.76 | 400.00 | +0.60 | +0.92 |
| **Train (HMOG)** | `backspace_rate` | 0.09 | 0.03 | 0.08 | 0.08 | 0.08 | 0.00 | 0.00 | 0.30 | +2.23 | +6.88 |
| **Test (BB-MAS)** | `backspace_rate` | 0.12 | 0.07 | 0.11 | 0.07 | 0.16 | 0.09 | 0.00 | 0.78 | +1.49 | +6.40 |
| **Train (HMOG)** | `contact_size_mean` | 0.0250 | 0.0022 | 0.0247 | 0.0235 | 0.0262 | 0.0027 | 0.0175 | 0.0363 | +0.77 | +0.97 |
| **Test (BB-MAS)** | `contact_size_mean` | 0.0276 | 0.0081 | 0.0258 | 0.0193 | 0.0363 | 0.0170 | 0.0175 | 0.0363 | 0.00 | -1.80 |
| **Train (HMOG)** | `touch_x_mean` | 634.45 | 186.35 | 616.09 | 549.58 | 685.04 | 135.47 | 49.00 | 1604.67 | +1.47 | +5.15 |
| **Test (BB-MAS)** | `touch_x_mean` | 783.70 | 336.07 | 864.87 | 600.44 | 1023.50 | 423.06 | 45.00 | 1361.74 | -0.67 | -0.60 |
| **Train (HMOG)** | `touch_y_mean` | 738.22 | 322.12 | 852.73 | 445.51 | 987.52 | 542.01 | 12.00 | 1458.00 | -0.58 | -0.98 |
| **Test (BB-MAS)** | `touch_y_mean` | 688.04 | 267.57 | 675.43 | 469.77 | 857.80 | 388.03 | 127.94 | 1924.05 | +0.81 | +0.91 |
| **Train (HMOG)** | `tap_accuracy` | 0.91 | 0.03 | 0.92 | 0.92 | 0.92 | 0.00 | 0.70 | 0.98 | -2.26 | +6.92 |
| **Test (BB-MAS)** | `tap_accuracy` | 0.88 | 0.07 | 0.89 | 0.84 | 0.93 | 0.09 | 0.60 | 0.98 | -1.00 | +1.58 |
| **Train (HMOG)** | `miss_rate` | 0.09 | 0.03 | 0.08 | 0.08 | 0.08 | 0.00 | 0.02 | 0.30 | +2.26 | +6.92 |
| **Test (BB-MAS)** | `miss_rate` | 0.12 | 0.07 | 0.11 | 0.07 | 0.16 | 0.09 | 0.02 | 0.40 | +1.00 | +1.58 |
| **Train (HMOG)** | `reaction_time_mean_ms` | 325.11 | 47.77 | 330.00 | 330.00 | 330.00 | 0.00 | 180.00 | 517.36 | -0.75 | +3.24 |
| **Test (BB-MAS)** | `reaction_time_mean_ms` | 307.70 | 56.14 | 313.15 | 270.50 | 348.05 | 77.55 | 180.00 | 450.36 | -0.29 | -0.38 |

*(Full CSV available at: [`paper_figures/eda/eda_summary_statistics_table.csv`](file:///C:/adavtive_dataset/paper_figures/eda/eda_summary_statistics_table.csv))*

---

## 3. Target UI Class Distribution (Train vs. Test)

| Target Class | UI Button Size | Train (`HMOG` - 7,194 Rows) | External Test (`BB-MAS` - 4,514 Rows) |
| :---: | :---: | :---: | :---: |
| **`0`** | **Small (40dp)** | 419 (`5.8%`) | 1,117 (`24.7%`) |
| **`1`** | **Default (48dp)** | 5,670 (`78.8%`) | 1,267 (`28.1%`) |
| **`2`** | **Large (56dp)** | 736 (`10.2%`) | 1,126 (`24.9%`) |
| **`3`** | **Extra Large (64dp)** | 369 (`5.1%`) | 1,004 (`22.2%`) |

---

## 4. Updated Publication Figures (300 DPI)

1. **[`fig1_feature_distributions.png`](file:///C:/adavtive_dataset/paper_figures/eda/fig1_feature_distributions.png)**: Multi-panel KDE and histogram overlay comparing Train (`HMOG`) and External Test (`BB-MAS`) distributions.
2. **[`fig2_outlier_boxplots.png`](file:///C:/adavtive_dataset/paper_figures/eda/fig2_outlier_boxplots.png)**: Side-by-side quartile dispersion and outlier boxplots across both datasets.
3. **[`fig3_context_impact_sitting_vs_walking.png`](file:///C:/adavtive_dataset/paper_figures/eda/fig3_context_impact_sitting_vs_walking.png)**: Split violin plots with Welch's two-sample $t$-test hypothesis annotations comparing sitting (`0`) vs. walking/moving (`1`).
4. **[`fig4_class_distribution_balance.png`](file:///C:/adavtive_dataset/paper_figures/eda/fig4_class_distribution_balance.png)**: Dual-panel bar chart quantifying target button size distribution in Train (`HMOG`) and External Test (`BB-MAS`).
5. **[`fig5_feature_correlation_matrix.png`](file:///C:/adavtive_dataset/paper_figures/eda/fig5_feature_correlation_matrix.png)**: Dual-panel Pearson correlation heatmaps showing structural alignment between Train and Test sets.
