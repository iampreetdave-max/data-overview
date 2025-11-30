import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Data-Science Analyzer", layout="wide")

st.markdown("""
    <style>
        .main { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); }
        .stTabs [data-baseweb="tab-list"] { background-color: #1e293b; }
        h1, h2, h3 { color: #e0e7ff; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_id_columns(df):
    """Auto-detect ID/index columns by name patterns."""
    id_patterns = ['id', 'ID', 'index', 'idx', 'pk', 'serial', 'code']
    id_cols = [col for col in df.columns if any(pat in col for pat in id_patterns)]
    return id_cols

def detect_types(df, feature_cols):
    """Auto-detect numeric vs categorical."""
    types = {}
    for col in feature_cols:
        try:
            pd.to_numeric(df[col], errors='coerce')
            numeric_ratio = df[col].apply(lambda x: isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.','').replace('-','').isdigit())).sum() / len(df)
            types[col] = 'numeric' if numeric_ratio > 0.8 else 'categorical'
        except:
            types[col] = 'categorical'
    return types

def preprocess_data(df, feature_cols, target_col, types):
    """Preprocess: impute, encode, scale."""
    df_clean = df.copy()
    
    # Handle missing values
    for col in feature_cols + [target_col]:
        if col in df_clean.columns:
            if types.get(col, 'categorical') == 'numeric':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown', inplace=True)
    
    # Encode categorical features
    le_dict = {}
    for col in feature_cols:
        if types.get(col, 'categorical') == 'categorical':
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            le_dict[col] = le
    
    # Convert target to numeric if needed
    if types.get(target_col, 'categorical') == 'categorical':
        le_target = LabelEncoder()
        df_clean[target_col] = le_target.fit_transform(df_clean[target_col].astype(str))
        task_type = 'classification'
    else:
        df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
        df_clean[target_col].fillna(df_clean[target_col].median(), inplace=True)
        task_type = 'regression'
    
    X = df_clean[feature_cols].values.astype(float)
    y = df_clean[target_col].values
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, task_type, le_dict

def compute_correlations(X, feature_cols):
    """Pearson correlation matrix."""
    df_X = pd.DataFrame(X, columns=feature_cols)
    return df_X.corr()

def compute_mutual_info(X, y, feature_cols, task_type):
    """Mutual information with target."""
    if task_type == 'classification':
        mi = mutual_info_classif(X, y, random_state=42)
    else:
        mi = mutual_info_regression(X, y, random_state=42)
    return dict(zip(feature_cols, mi))

def train_models(X, y, task_type):
    """Train multiple models with cross-validation."""
    if task_type == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Ridge Classifier': LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42),
        }
        scoring = {'accuracy': 'accuracy', 'precision': 'precision_weighted', 'recall': 'recall_weighted', 'f1': 'f1_weighted', 'roc_auc': 'roc_auc_ovr_weighted'}
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=0.1, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            'SVR': SVR(kernel='rbf'),
            'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
        }
        scoring = {'mse': 'neg_mean_squared_error', 'mae': 'neg_mean_absolute_error', 'r2': 'r2'}
    
    results = {}
    for name, model in models.items():
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
        results[name] = cv_results
    
    return results, models

def compute_feature_importance(X, y, feature_cols, models, task_type):
    """Feature importance from multiple sources."""
    importance = {col: [] for col in feature_cols}
    
    # 1. Correlation (for numeric y)
    if task_type == 'regression':
        corr = np.corrcoef(X.T, y)[:-1, -1]
        for col, c in zip(feature_cols, np.abs(corr)):
            importance[col].append(c)
    
    # 2. Mutual Information
    mi = compute_mutual_info(X, y, feature_cols, task_type)
    mi_norm = max(mi.values()) if mi else 1
    for col in feature_cols:
        importance[col].append(mi[col] / (mi_norm + 1e-8))
    
    # 3. Tree-based model importances
    for name, model in models.items():
        if hasattr(model, 'fit'):
            model.fit(X, y)
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                for col, i in zip(feature_cols, imp):
                    importance[col].append(i)
    
    # Average across sources
    avg_importance = {col: np.mean(scores) if scores else 0 for col, scores in importance.items()}
    return sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

def detect_anomalies(X, feature_cols, method='isolation_forest'):
    """Detect outliers using IQR method."""
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(contamination=0.05, random_state=42)
    anomaly_labels = iso.fit_predict(X)
    anomaly_indices = np.where(anomaly_labels == -1)[0]
    return anomaly_indices

def generate_shap_explanation(X, y, model, feature_cols, num_samples=100):
    """Generate SHAP explanations."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:num_samples])
        return explainer, shap_values
    except:
        return None, None

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("🔬 Data-Science Analyzer")
st.write("Upload CSV → Auto-detect types → Train multiple models → Deep analysis")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type='csv')

if uploaded_file is None:
    st.info("👈 Upload a CSV file to begin analysis")
    st.stop()

# Load data
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

st.success(f"✓ Loaded {df.shape[0]} rows × {df.shape[1]} columns")

# ============================================================================
# SECTION A: DATA OVERVIEW & ID DETECTION
# ============================================================================

with st.expander("📋 Section A: Data Overview", expanded=True):
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Auto-detect ID columns
    id_cols = detect_id_columns(df)
    feature_cols_auto = [col for col in df.columns if col not in id_cols]
    
    if id_cols:
        st.warning(f"🔑 Auto-detected ID columns (excluded from analysis): {', '.join(id_cols)}")
    
    st.write("**Data Preview:**")
    st.dataframe(df.head(), use_container_width=True)
    
    st.write("**Data Types & Missing Values:**")
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Missing': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(info_df, use_container_width=True)

# ============================================================================
# SELECT TARGET & FEATURES
# ============================================================================

st.divider()
col1, col2 = st.columns([1, 2])

with col1:
    target_col = st.selectbox("🎯 Select target column:", feature_cols_auto)

with col2:
    selected_features = st.multiselect(
        "📊 Select features (leave empty for all):",
        feature_cols_auto,
        default=feature_cols_auto[:min(10, len(feature_cols_auto))]
    )

feature_cols = selected_features if selected_features else feature_cols_auto
feature_cols = [col for col in feature_cols if col != target_col]  # Exclude target

if not feature_cols:
    st.error("❌ No features selected. Please select at least one feature.")
    st.stop()

# ============================================================================
# DETECT TYPES & PREPROCESS
# ============================================================================

types = detect_types(df, feature_cols + [target_col])

try:
    X, y, task_type, le_dict = preprocess_data(df, feature_cols, target_col, types)
except Exception as e:
    st.error(f"Preprocessing error: {e}")
    st.stop()

st.write(f"**Task Type Detected:** {task_type.upper()}")

# ============================================================================
# SECTION C: MODEL TRAINING & COMPARISON
# ============================================================================

st.divider()
st.header("🤖 Section C: Model Training & Comparison")

with st.spinner("⏳ Training models (5-fold CV)..."):
    results, models = train_models(X, y, task_type)

# Display results
if task_type == 'classification':
    st.subheader("Classification Models - Cross-Validation Scores")
    
    comparison_df = []
    for model_name, cv_results in results.items():
        comparison_df.append({
            'Model': model_name,
            'Accuracy': f"{cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}",
            'Precision': f"{cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}",
            'Recall': f"{cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}",
            'F1': f"{cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}",
            'ROC-AUC': f"{cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}",
        })
    
    comparison_df = pd.DataFrame(comparison_df).sort_values('ROC-AUC', ascending=False)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visualize model comparison
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        acc_scores = [results[m]['test_accuracy'].mean() for m in results.keys()]
        models_list = list(results.keys())
        ax.barh(models_list, acc_scores, color='steelblue')
        ax.set_xlabel('Accuracy')
        ax.set_title('Model Accuracy Comparison')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        auc_scores = [results[m]['test_roc_auc'].mean() for m in results.keys()]
        ax.barh(models_list, auc_scores, color='coral')
        ax.set_xlabel('ROC-AUC')
        ax.set_title('Model ROC-AUC Comparison')
        plt.tight_layout()
        st.pyplot(fig)

else:  # Regression
    st.subheader("Regression Models - Cross-Validation Scores")
    
    comparison_df = []
    for model_name, cv_results in results.items():
        rmse = np.sqrt(-cv_results['test_mse'].mean())
        mae = -cv_results['test_mae'].mean()
        r2 = cv_results['test_r2'].mean()
        comparison_df.append({
            'Model': model_name,
            'RMSE': f"{rmse:.4f} ± {np.sqrt(cv_results['test_mse'].std()):.4f}",
            'MAE': f"{mae:.4f} ± {-cv_results['test_mae'].std():.4f}",
            'R²': f"{r2:.4f} ± {cv_results['test_r2'].std():.4f}",
        })
    
    comparison_df = pd.DataFrame(comparison_df).sort_values('R²', ascending=False)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visualize
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        rmse_scores = [np.sqrt(-results[m]['test_mse'].mean()) for m in results.keys()]
        models_list = list(results.keys())
        ax.barh(models_list, rmse_scores, color='steelblue')
        ax.set_xlabel('RMSE (lower is better)')
        ax.set_title('Model RMSE Comparison')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        r2_scores = [results[m]['test_r2'].mean() for m in results.keys()]
        ax.barh(models_list, r2_scores, color='coral')
        ax.set_xlabel('R² Score')
        ax.set_title('Model R² Comparison')
        plt.tight_layout()
        st.pyplot(fig)

# ============================================================================
# SECTION D: FEATURE IMPORTANCE
# ============================================================================

st.divider()
st.header("⭐ Section D: Feature Importance & Selection")

# Train best model for SHAP
best_model_name = list(results.keys())[0]
best_model = models[best_model_name]
best_model.fit(X, y)

importance_ranking = compute_feature_importance(X, y, feature_cols, models, task_type)

st.subheader("Top Features by Importance")
importance_df = pd.DataFrame(importance_ranking, columns=['Feature', 'Importance Score'])
st.dataframe(importance_df.head(10), use_container_width=True)

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
top_n = min(10, len(importance_ranking))
features = [f[0] for f in importance_ranking[:top_n]]
scores = [f[1] for f in importance_ranking[:top_n]]
ax.barh(features, scores, color='mediumseagreen')
ax.set_xlabel('Importance Score')
ax.set_title(f'Top {top_n} Features')
plt.tight_layout()
st.pyplot(fig)

st.info(f"💡 **Recommended feature subset:** Use top 5-7 features: {', '.join([f[0] for f in importance_ranking[:7]])}")

# ============================================================================
# SECTION E: SHAP EXPLANATIONS
# ============================================================================

st.divider()
st.header("🎯 Section E: Model Explanations (SHAP)")

try:
    explainer, shap_values = generate_shap_explanation(X, y, best_model, feature_cols, num_samples=min(100, X.shape[0]))
    
    if explainer is not None:
        st.write(f"Using **{best_model_name}** for SHAP explanations")
        
        # Global explanation
        st.subheader("Global Feature Importance (SHAP Mean |values|)")
        fig = plt.figure(figsize=(10, 6))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], X[:min(100, X.shape[0])], feature_names=feature_cols, show=False, plot_type='bar')
        else:
            shap.summary_plot(shap_values, X[:min(100, X.shape[0])], feature_names=feature_cols, show=False, plot_type='bar')
        st.pyplot(fig)
        plt.close()
        
        # Local explanation (first sample)
        st.subheader("Local Explanation - Sample #0")
        fig = plt.figure(figsize=(10, 6))
        if isinstance(shap_values, list):
            shap.force_plot(explainer.expected_value[0], shap_values[0][0], X[0], feature_names=feature_cols, matplotlib=True, show=False)
        else:
            shap.force_plot(explainer.expected_value, shap_values[0], X[0], feature_names=feature_cols, matplotlib=True, show=False)
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("⚠️ SHAP explanations not available for this model type.")
except Exception as e:
    st.warning(f"⚠️ Could not generate SHAP explanations: {e}")

# ============================================================================
# SECTION B: DESCRIPTIVE ANALYSIS
# ============================================================================

st.divider()
st.header("📊 Section B: Descriptive Analysis & Patterns")

# Correlations
st.subheader("Feature Correlations")
corr_matrix = compute_correlations(X, feature_cols)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
plt.title('Correlation Matrix')
plt.tight_layout()
st.pyplot(fig)

# Mutual Information
st.subheader("Mutual Information with Target")
mi_scores = compute_mutual_info(X, y, feature_cols, task_type)
mi_df = pd.DataFrame(sorted(mi_scores.items(), key=lambda x: x[1], reverse=True), columns=['Feature', 'MI Score'])
st.dataframe(mi_df, use_container_width=True)

# Anomaly Detection
st.subheader("Anomaly Detection")
anomalies = detect_anomalies(X, feature_cols)
st.write(f"**Detected {len(anomalies)} anomalies (~{len(anomalies)/len(X)*100:.2f}% of data)**")
if len(anomalies) > 0:
    st.write(f"Anomaly indices: {anomalies[:20]}...")

# Feature Statistics
st.subheader("Feature Statistics (numeric only)")
numeric_cols = [col for col in feature_cols if types[col] == 'numeric']
if numeric_cols:
    stats_df = df[numeric_cols].describe().T
    st.dataframe(stats_df, use_container_width=True)

# ============================================================================
# SECTION F: RECOMMENDED PIPELINE
# ============================================================================

st.divider()
st.header("✅ Section F: Recommended Final Pipeline")

best_model_for_pipeline = sorted(results.items(), key=lambda x: x[1]['test_roc_auc' if task_type == 'classification' else 'test_r2'].mean(), reverse=True)[0]

st.subheader(f"Recommended Model: {best_model_for_pipeline[0]}")
if task_type == 'classification':
    best_score = best_model_for_pipeline[1]['test_roc_auc'].mean()
    st.metric("Expected ROC-AUC (CV)", f"{best_score:.4f}")
else:
    best_score = best_model_for_pipeline[1]['test_r2'].mean()
    st.metric("Expected R² (CV)", f"{best_score:.4f}")

st.code(f"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier  # or {best_model_for_pipeline[0]}
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. Load & Preprocess
df = pd.read_csv('your_data.csv')
X = df[{feature_cols[:7]}]  # Top 7 features recommended
y = df['{target_col}']

# Handle missing values
X.fillna(X.median(), inplace=True)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Train Model
model = {best_model_for_pipeline[0]}(random_state=42, n_jobs=-1)
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')

# 3. Final model
model.fit(X_scaled, y)
predictions = model.predict(X_scaled)

print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
""", language='python')

st.success("✓ Analysis complete! Download results or adjust parameters above.")
