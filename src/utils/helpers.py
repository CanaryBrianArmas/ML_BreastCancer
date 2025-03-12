import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def load_data(sample_path):
    """Load sample dataset"""
    return pd.read_csv(sample_path)


def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }
    
    if y_proba is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_proba)
    
    return metrics


def create_pipeline(classifier):
    """Create preprocessing and modeling pipeline with imputation"""
    
     # 1. Imputación solo para la columna problemática
    preprocessor = ColumnTransformer(
        transformers=[
            ('imputer', SimpleImputer(strategy='median'), ['Bare.nuclei'])  # Columna con nulos
        ],
        remainder='passthrough'  # Pasa el resto de columnas sin cambios
    )
    
    # 2. Pipeline completa
    return Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),  # Escala todas las columnas
        ('classifier', classifier)
    ])