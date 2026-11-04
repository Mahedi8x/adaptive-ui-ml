# 📱 Context-Aware Adaptive User Interface (AUI) Machine Learning Framework

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost Champion](https://img.shields.io/badge/Model-XGBoost%20(95.24%25%20Acc)-brightgreen.svg)]()
[![Explainability](https://img.shields.io/badge/XAI-SHAP%20Interpretability-orange.svg)]()
[![Domain](https://img.shields.io/badge/HCI-Human--Computer%20Interaction-purple.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, context-aware Machine Learning framework that dynamically personalizes and scales mobile user interface elements (Button geometries: `Small`, `Medium`, `Large`, `Extra_Large`) based on real-time Human-Computer Interaction (HCI) behavioral metrics, situational context, and user motor control dynamics.

---

## 📌 Executive Summary & Research Motivation

Standard mobile user interfaces adopt a "one-size-fits-all" approach, leading to high miss-tap rates and elevated user cognitive load during movement (e.g., walking vs. sitting) or for users with diverse motor dexterity.

This project delivers an end-to-end Machine Learning pipeline that:
1. **Predicts Optimal Target Sizes** based on tactile interaction dynamics (reaction time, miss-tap rates, tap accuracy, inter-finger intervals).
2. **Mitigates Cold-Start Latency** by providing instant heuristics before user session behavioral convergence.
3. **Provides Explainable Decisions (XAI)** through SHAP (SHapley Additive exPlanations) analysis to demystify HCI feature importance.

---

## 🔬 Benchmark Model Comparison

Extensive comparative benchmarking across classic, ensemble, and neural classifiers:

| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Status |
|---|---|---|---|---|---|
| **Tuned XGBoost Classifier** | **95.24%** | **0.952** | **0.952** | **0.952** | 🏆 **Champion** |
| Random Forest (100 Trees) | 91.80% | 0.919 | 0.918 | 0.918 | Baseline |
| Decision Tree (CART) | 87.50% | 0.876 | 0.875 | 0.875 | Baseline |
| Support Vector Machine (RBF) | 82.35% | 0.824 | 0.823 | 0.823 | Baseline |
| Logistic Regression | 74.30% | 0.742 | 0.743 | 0.742 | Baseline |

> Tuned via 5-Fold Stratified Cross-Validation with `GridSearchCV` on Bayesian hyperparameter spaces.

---

## 📊 Key HCI Behavioral Features

The model learns from dynamic interaction parameters:
- **`tap_accuracy`**: Ratio of successful hits to total touch attempts (`successful_taps / total_taps`).
- **`avg_reaction_time_ms`**: Latency between target presentation and touch event.
- **`miss_rate`**: Percentage of taps landing outside target bounding boxes.
- **`IFI` (Inter-Finger Interval)**: Temporal spacing between consecutive touch gestures.
- **`screen_dimensions`**: Device viewport geometry (`screen_width`, `screen_height`).
- **`contextual_state`**: User situational context (Static / Sitting vs. Dynamic / Walking).

---

## 🔍 Explainable AI (SHAP Analysis)

To guarantee safety in assistive interfaces, we employ **SHAP TreeExplainer**:
- Identifies non-linear interactions between tap velocity and error rates.
- Demonstrates that `avg_reaction_time_ms` and `miss_rate` contribute over **68% of feature attribution** when scaling buttons from `Medium` to `Extra_Large`.

---

## 🚀 Quickstart & Pipeline Execution

### 1. Installation
```bash
git clone https://github.com/AdaptiveUIReseach/new_data.git
cd new_data
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis & Benchmark
```bash
python run_eda.py
python run_cold_start_shap.py
```

### 3. Inference with Pre-trained Pipeline
```python
import joblib
import pandas as pd

# Load serialized artifacts
model = joblib.load("saved_models/XGBoost.pkl")
scaler = joblib.load("saved_models/feature_scaler.pkl")

# Sample HCI incoming telemetrics
sample_user = pd.DataFrame([{
    "age": 28,
    "screen_width": 1080,
    "screen_height": 2400,
    "total_taps": 15,
    "successful_taps": 11,
    "miss_taps": 4,
    "tap_accuracy": 0.733,
    "avg_reaction_time_ms": 420.5,
    "miss_rate": 0.267,
    "IFI": 0.65
}])

# Predict adaptive UI size class
prediction = model.predict(scaler.transform(sample_user))
print("Recommended Button Size:", prediction[0])
```

---

## 📂 Project Architecture

```text
adavtive_model/
├── paper_figures/                  # Publication-grade figures & charts
│   ├── cold_start_shap/            # SHAP summary & dependence plots
│   └── eda/                        # Feature distribution & correlation heatmaps
├── saved_models/                   # Exported serializations (.pkl & .json)
│   ├── xgboost_champion.json       # Top-performing XGBoost model
│   └── feature_scaler.pkl          # Pre-fitted StandardScaler
├── 01_exploratory_data_analysis.ipynb
├── 02_cold_start_ml_and_shap.ipynb
├── run_cold_start_shap.py          # Standalone reproducible execution script
├── run_eda.py                      # Data validation & summary stats
├── requirements.txt                # Pinned dependencies
└── README.md                       # Documentation
```

---

## 👨‍💻 Research & Development

**MD Mohiuddin Mahedi**  
- GitHub: [@mehedi2351](https://github.com/mehedi2351)  
- Email: mohiuddinmahadi08@gmail.com  
- Research Domain: Human-Computer Interaction (HCI), Applied ML, Adaptive Interfaces
