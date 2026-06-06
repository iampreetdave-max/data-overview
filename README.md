# Data-Science Analyzer

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

Upload a CSV and get an automated, end-to-end machine-learning analysis: type detection, model benchmarking, feature importance, and explainability — all from a single Streamlit app.

## Overview

Data-Science Analyzer is a self-contained Streamlit application that turns a raw CSV into a complete supervised-learning report without writing any code. You upload a dataset, pick a target column, and the app infers whether the problem is classification or regression, preprocesses the data, trains and cross-validates a suite of models, ranks features by importance, and produces SHAP explanations and descriptive diagnostics.

It is built for fast exploratory modelling — letting you compare algorithms and understand what drives a target variable in a few clicks.

## Key Features

- **CSV upload** with automatic row/column/memory summary and a data-preview table.
- **Automatic ID-column detection** (by name patterns such as `id`, `index`, `pk`, `code`) so identifier columns are excluded from modelling.
- **Automatic type inference** — numeric vs. categorical — driving downstream preprocessing.
- **Preprocessing pipeline**: median/mode imputation for missing values, label encoding for categoricals, and standard scaling of features.
- **Task auto-detection**: classification when the target is categorical, regression when numeric.
- **Multi-model benchmarking with 5-fold cross-validation**:
  - Classification: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVM, K-Nearest Neighbors, Ridge Classifier.
  - Regression: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, SVR, K-Nearest Neighbors.
- **Scored comparison tables and bar charts** — accuracy, precision, recall, F1, ROC-AUC for classification; RMSE, MAE, R² for regression.
- **Aggregated feature importance** combining correlation, mutual information, and tree-based importances.
- **SHAP explanations** with global summary and local force plots for tree-based models.
- **Descriptive analysis**: correlation heatmap, mutual-information ranking, Isolation-Forest anomaly detection, and numeric feature statistics.
- **Recommended pipeline**: surfaces the best-scoring model and emits a ready-to-use code snippet.

## How It Works

The app (`app.py`) is organized as a linear Streamlit flow backed by helper functions:

1. **Ingest** — read the uploaded CSV into a pandas DataFrame.
2. **Detect** — identify ID columns and infer per-column numeric/categorical types.
3. **Preprocess** — impute, label-encode, and scale; decide classification vs. regression from the target.
4. **Train** — run each candidate model through cross-validation and collect scores.
5. **Explain** — compute aggregated feature importance and SHAP values for the top model.
6. **Diagnose** — render correlations, mutual information, anomalies, and summary statistics.
7. **Recommend** — pick the highest-scoring model and generate a reproducible pipeline snippet.

## Tech Stack

- **Language:** Python 3
- **App framework:** Streamlit
- **ML / data:** scikit-learn, XGBoost, SHAP, NumPy, pandas, SciPy
- **Visualization:** Matplotlib, Seaborn

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/iampreetdave-max/data-overview.git
cd data-overview
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (default http://localhost:8501) and upload a CSV to begin.

## Project Structure

```
data-overview/
├── app.py             # Streamlit app: preprocessing, training, explainability
├── requirements.txt   # Pinned dependencies
└── README.md
```

## Notes

- Heavier models (SVM/SVR, gradient boosting) on large datasets will increase training time, since every model is cross-validated.
- SHAP force/summary plots are generated for tree-based models; other model types fall back gracefully.
