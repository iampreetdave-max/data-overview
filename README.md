# Data-Science Analyzer

An interactive Streamlit application that turns a raw CSV into a full supervised-learning analysis: automatic type detection, multi-model benchmarking, feature importance, and SHAP explanations.

## Overview

Data-Science Analyzer is a single-page Streamlit app that runs an end-to-end tabular modeling workflow from a file upload. A user uploads a CSV; the app profiles the data, detects and excludes ID-like columns, infers whether each feature is numeric or categorical, and determines from the chosen target whether the task is classification or regression. It then preprocesses the data, trains a panel of models under 5-fold cross-validation, ranks features by importance, and produces SHAP explanations and descriptive diagnostics.

The app is self-contained in `app.py` and is deployable as-is to Streamlit Community Cloud or any Streamlit-capable host; the pinned `requirements.txt` defines the full runtime.

## Key Features

- CSV upload with automatic data profiling: row/column counts, memory footprint, dtypes, and missing-value percentages.
- Automatic ID-column detection (by name patterns) and exclusion from modeling.
- Automatic numeric-vs-categorical type inference and task detection (classification vs regression).
- Preprocessing pipeline: median/mode imputation, label encoding of categoricals, and standardization.
- Multi-model training under 5-fold cross-validation:
  - Classification: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVM, K-Nearest Neighbors, and an L2 (ridge) logistic variant.
  - Regression: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, SVR, and K-Nearest Neighbors.
- Model comparison tables and bar charts (accuracy/ROC-AUC for classification; RMSE/R-squared for regression).
- Feature importance aggregated across correlation, mutual information, and tree-based model importances.
- SHAP global and local explanations for the top-ranked model.
- Descriptive analysis: correlation heatmap, mutual information with the target, Isolation Forest anomaly detection, and numeric feature statistics.
- A generated, copy-ready scikit-learn pipeline snippet for the recommended model and feature subset.

## How It Works

`app.py` is organized into helper functions and a top-to-bottom Streamlit flow:

1. Upload and profile the CSV (Section A).
2. Select target and features; infer column types.
3. `preprocess_data` imputes, encodes, and scales; the task type is derived from the target.
4. `train_models` runs cross-validation across the model panel for the detected task.
5. `compute_feature_importance` blends correlation, mutual information, and model-derived importances.
6. `generate_shap_explanation` builds SHAP values for the leading model.
7. Descriptive diagnostics and a recommended pipeline are rendered last.

## Tech Stack

- Language: Python 3
- App framework: Streamlit
- Modeling: scikit-learn, XGBoost
- Explainability: SHAP
- Data and plotting: pandas, NumPy, SciPy, matplotlib, seaborn

## Getting Started

### Prerequisites

- Python 3.9 or newer
- A tabular CSV with a column suitable as a prediction target

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

Open the local URL Streamlit prints, upload a CSV, choose a target column, and review the generated analysis.

## Configuration

No environment variables or API keys are required. Runtime behavior is controlled entirely through the UI (target column, feature selection). Dependency versions are pinned in `requirements.txt`.

## Project Structure

```
data-overview/
├── app.py             # Streamlit application (full analysis workflow)
├── requirements.txt   # Pinned runtime dependencies
└── README.md
```
