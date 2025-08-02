"""
Utility functions for MLOps pipeline
"""
import numpy as np
import joblib
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path("models")

def load_california_housing_data(test_size=0.2, random_state=42):
    """Load and preprocess dataset"""
    housing = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        housing.data, housing.target, 
        test_size=test_size, 
        random_state=random_state
    )
    scaler = StandardScaler()
    return (
        scaler.fit_transform(X_train),
        scaler.transform(X_test),
        y_train,
        y_test,
        scaler
    )

def calculate_mse(y_true, y_pred):
    """Calculate Mean Squared Error"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)
def save_model_artifacts(model, scaler):
    """Save model artifacts with error handling"""
    try:
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_DIR/"linear_regression.joblib")
        joblib.dump(scaler, MODEL_DIR/"scaler.joblib")
    except Exception as e:
        raise RuntimeError(f"Failed to save artifacts: {str(e)}")

def load_model_artifacts():
    """Load model artifacts with validation"""
    try:
        model = joblib.load(MODEL_DIR/"linear_regression.joblib")
        scaler = joblib.load(MODEL_DIR/"scaler.joblib")
        assert hasattr(model, 'coef_'), "Invalid model file"
        assert hasattr(scaler, 'transform'), "Invalid scaler file"
        return model, scaler
    except FileNotFoundError:
        raise FileNotFoundError("Model not found. Train model first.")