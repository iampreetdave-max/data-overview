# Data Overview

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EC6C35?style=flat)

> A Streamlit web app for interactive dataset exploration — upload any CSV and get instant statistics, visualizations, ML model training, and SHAP explanations.

## About

Data Overview is an all-in-one data analysis dashboard built with Streamlit. Upload any CSV file and instantly explore it with summary statistics, distribution plots, correlation heatmaps, and automated ML model training (XGBoost, RandomForest, etc.) with SHAP feature importance explanations. Designed for quick EDA and model prototyping without writing code.

## Tech Stack

- **Language:** Python 3
- **Dashboard:** Streamlit
- **Data:** Pandas, NumPy
- **ML:** scikit-learn, XGBoost
- **Explainability:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **Stats:** SciPy

## Features

- **CSV upload** — drag and drop any dataset
- **Summary statistics** — shape, dtypes, missing values, descriptive stats
- **Distribution plots** — histograms and KDE for all numeric columns
- **Correlation heatmap** — visualize feature relationships
- **Automated ML training** — train models on your data with target column selection
- **SHAP explanations** — understand which features drive predictions
- **Interactive UI** — filter, sort, and explore data in the browser

## Getting Started

### Prerequisites

- Python 3.8+

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

Open the URL shown in the terminal (typically [http://localhost:8501](http://localhost:8501)).

## Project Structure

```
data-overview/
├── app.py              # Streamlit dashboard
├── requirements.txt    # Python dependencies
└── README.md
```

## License

This project is open source.
