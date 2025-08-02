"""
Prediction script for model verification
"""
import numpy as np
from utils import load_model_artifacts, load_california_housing_data


def make_predictions():
    """
    Load trained model and make predictions on test set
    """
    print("Loading trained model...")
    
    try:
        model, scaler = load_model_artifacts()
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: Model files not found. Please run training first.")
        return
    
    # Load test data
    print("Loading test data...")
    _, X_test, _, y_test, _ = load_california_housing_data()
    
    print(f"Test set size: {X_test.shape}")
    print(f"Number of features: {X_test.shape[1]}")
    
    # Make predictions on the entire test set
    print("Making predictions...")
    predictions = model.predict(X_test)
    
    # Calculate some basic statistics
    print(f"\nPrediction Statistics:")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Prediction range: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
    print(f"Mean prediction: {np.mean(predictions):.4f}")
    print(f"Std prediction: {np.std(predictions):.4f}")
    
    # Show sample predictions vs actual values
    print(f"\nSample Predictions (first 10):")
    print("Predicted\tActual\t\tDifference")
    print("-" * 40)
    for i in range(10):
        diff = predictions[i] - y_test[i]
        print(f"{predictions[i]:.4f}\t\t{y_test[i]:.4f}\t\t{diff:.4f}")
    
    # Calculate mean absolute error for the sample
    mae = np.mean(np.abs(predictions - y_test))
    print(f"\nMean Absolute Error on test set: {mae:.4f}")
    
    # Model coefficients info
    print(f"\nModel Information:")
    print(f"Number of coefficients: {len(model.coef_)}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Coefficient range: [{np.min(model.coef_):.4f}, {np.max(model.coef_):.4f}]")
    
    print("\nPrediction completed successfully!")
    
    return predictions


if __name__ == "__main__":
    make_predictions()