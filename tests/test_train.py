"""
Unit tests for model training pipeline
"""
import os
import pytest
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_california_housing_data, calculate_mse, load_model_artifacts
from train import train_model, MODEL_DIR

@pytest.fixture(scope="module")
def trained_model():
    """Shared test fixture - trains model once"""
    return train_model()

class TestDataLoading:
    def test_dataset_loading(self):
        X_train, X_test, _, _, _ = load_california_housing_data()
        assert X_train.shape[1] == X_test.shape[1] == 8

class TestModelTraining:
    def test_model_properties(self, trained_model):
        model, _, _ = trained_model
        assert isinstance(model, LinearRegression)
        assert len(model.coef_) == 8
        assert isinstance(model.intercept_, float)

class TestModelPerformance:
    def test_r2_threshold(self, trained_model):
        _, _, test_r2 = trained_model
        assert test_r2 >= 0.5
    
    @pytest.mark.parametrize("y_true,y_pred,expected", [
        ([1, 2, 3], [1, 2, 3], 0.0),        # Perfect prediction
        ([1, 2, 3], [2, 3, 4], 1.0),        # Constant offset
        ([0, 0], [1, -1], 1.0)              # Mixed errors
    ])
    def test_mse_calculation(self, y_true, y_pred, expected):
        assert np.isclose(calculate_mse(y_true, y_pred), expected)

class TestArtifacts:
    def test_artifact_creation(self, trained_model):
        assert (MODEL_DIR/"linear_regression.joblib").exists()
        assert (MODEL_DIR/"scaler.joblib").exists()
    
    def test_artifact_loading(self, trained_model):
        model, scaler = load_model_artifacts()
        assert model.coef_.shape == (8,)
        assert callable(scaler.transform)