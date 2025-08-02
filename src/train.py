"""
Model training script for Linear Regression on California Housing dataset
"""
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from utils import load_california_housing_data, calculate_mse, save_model_artifacts

MODEL_DIR = Path("models")

def train_model():
    """
    Train Linear Regression model on California Housing dataset
    Returns:
        tuple: (model, scaler, test_r2_score)
    """
    X_train, X_test, y_train, y_test, scaler = load_california_housing_data()
    
    print(f"Dataset Shapes:")
    print(f"Train features: {X_train.shape}, Test features: {X_test.shape}")
    print(f"Train targets: {y_train.shape}, Test targets: {y_test.shape}")
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics (REQUIRED BY ASSIGNMENT)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = calculate_mse(y_train, y_train_pred)
    test_mse = calculate_mse(y_test, y_test_pred)
    
    # Print results (EXPLICITLY REQUIRED)
    print("\nTraining Metrics:")
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Training MSE (Loss): {train_mse:.4f}")
    print(f"\nTest Metrics:")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Test MSE (Loss): {test_mse:.4f}")
    
    # Save artifacts
    MODEL_DIR.mkdir(exist_ok=True)
    save_model_artifacts(model, scaler)
    
    # Print quantization-ready parameters
    print("\nModel Parameters for Quantization:")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_:.4f}")
    
    return model, scaler, test_r2

if __name__ == "__main__":
    train_model()