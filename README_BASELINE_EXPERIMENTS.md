# Empirical Evaluation & Baseline Benchmarking Report
## Phase 1: Biomechanical Data Preprocessing & Multi-Model Cold-Start Classification

> **Project:** Adaptive Smartphone UI Framework for Cross-Application Optimization  
> **Authors:** Research Team  
> **Target Venue:** IEEE Transactions on Human-Machine Systems / ACM TiiS / Applied Soft Computing  
> **Associated Notebook:** `01_preprocessing_and_baseline_benchmarking.ipynb`

---

## 1. Experimental Overview & Scientific Rationale

In cold-start UI adaptation, when a new user interacts with a smartphone application, the operating system possesses zero historical personalization data. To bypass erratic trial-and-error exploration, our framework leverages a population-level supervised classification prior.

This phase systematically compares **5 distinct machine learning model families** under a strict **5-Fold GroupKFold Cross-Validation** strategy partitioned exclusively by `user_id` (141 unique subjects from HMOG and Touchalytics). This guarantees that training and validation folds never share samples from the same human user, thereby mathematically preventing **Data Leakage**.

---

## 2. Table 1: Multi-Model Benchmark Comparison (User-Split GroupKFold)

The table below provides the full comparative evaluation across the 5 benchmarked families:

| Algorithm Family | Model | Accuracy (%) | Precision (%) | Recall (%) | Macro F1 (%) | Weighted F1 (%) | Inference Latency (ms/sample) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ensemble (Bagging)** | **Random Forest** | **99.95** | **99.95** | **99.95** | **99.91** | **99.95** | 0.0036 ms |
| **Rule-Based Tree** | Decision Tree (CART) | 99.93 | 99.93 | 99.93 | 99.90 | 99.93 | **0.0001 ms** |
| **Ensemble (Boosting)** | **XGBoost (Proposed)** | 99.57 | 99.59 | 99.57 | 98.80 | 99.57 | 0.0013 ms |
| **Kernel Machine** | Support Vector Machine (RBF) | 98.24 | 98.31 | 98.24 | 95.96 | 98.24 | 0.0305 ms |
| **Linear Model** | Multinomial Logistic Regression | 97.75 | 97.80 | 97.75 | 95.54 | 97.75 | 0.0002 ms |

*(File saved in: `paper_figures/baseline_model_comparison_table.csv`)*

---

## 3. Key Findings & Discussion for Paper

1. **Superiority of Non-Linear Boundary Modeling:**
   Tree-based ensemble models (Random Forest and XGBoost) significantly outperformed linear baselines (Logistic Regression at 97.75% and Linear/RBF SVM at 98.24%). Touch biomechanics (such as the non-linear interaction between swipe velocity and finger contact area) are modeled more accurately via split-entropy trees.

2. **Ultra-Low On-Device Latency:**
   For mobile OS integration, execution latency is critical to prevent UI frame drops (which must stay below the 16.6 ms budget for 60 Hz displays). All tested models exhibited per-sample inference latencies under **0.035 ms**, making them exceptionally lightweight and feasible for real-time mobile deployment without battery or CPU strain.

3. **Trade-off Between Random Forest and XGBoost:**
   While Random Forest achieved marginally higher F1 (99.95%), XGBoost offers native single-file JSON model export (`cold_start_xgboost.json`) with standardized C++/Kotlin tree traversal, making XGBoost the ideal engine for real-world Android OS deployment.

---

## 4. Generated Publication Visualizations

All publication-ready figures have been generated at 300 DPI in the `paper_figures/` folder:

1. **`paper_figures/feature_correlation_heatmap.png`**:
   Visualizes the Pearson correlation across all 11 biomechanical features (demonstrating independence between swipe velocity, contact area, and normalized reaction time).
2. **`paper_figures/model_comparison_barchart.png`**:
   Side-by-side grouped bar plot highlighting the Accuracy and Weighted F1-Score across all 5 benchmarked models.
3. **`paper_figures/best_model_confusion_matrix.png`**:
   Heatmap confusion matrix demonstrating near-zero misclassifications across the 4 discrete UI button size targets (`Small [40dp]`, `Default [48dp]`, `Large [56dp]`, and `Extra Large [64dp]`).

---

## 5. Reproducibility Instructions

To replicate the entire experimental pipeline from scratch:
```bash
# 1. Open the interactive Jupyter Notebook
jupyter notebook 01_preprocessing_and_baseline_benchmarking.ipynb

# 2. Or execute via headless terminal
python -m jupyter nbconvert --to notebook --execute --inplace 01_preprocessing_and_baseline_benchmarking.ipynb
```
