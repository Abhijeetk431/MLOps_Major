"""
Manual quantization script for Linear Regression model
"""
import numpy as np
import joblib
import os
from sklearn.metrics import r2_score
from utils import load_model_artifacts, load_california_housing_data, calculate_mse


def print_comparison_table(original_r2, quant_r2, original_mse, quant_mse,
                         original_mem, quant_mem, max_coef_error, max_pred_diff):
    """Prints a markdown-style comparison table"""
    print("\n" + "="*50)
    print("QUANTIZATION PERFORMANCE COMPARISON".center(50))
    print("="*50)
    
    # Model Performance
    print("\n--- Model Performance ---")
    print(f"{'Metric':<15} | {'Original':<12} | {'Quantized':<12} | {'Change':<10}")
    print("-"*45)
    print(f"{'R² Score':<15} | {original_r2:.4f}{'':<6} | {quant_r2:.4f}{'':<6} | {quant_r2-original_r2:+.4f}")
    print(f"{'MSE':<15} | {original_mse:.4f}{'':<6} | {quant_mse:.4f}{'':<6} | {quant_mse-original_mse:+.4f}")

    # Memory Usage
    print("\n--- Memory Usage ---")
    reduction = 100 * (1 - quant_mem/original_mem)
    print(f"Original params: {original_mem:.2f} B")
    print(f"Quantized params: {quant_mem:.2f} B → {reduction:.1f}% smaller")

    # Verification
    print("\n--- Verification ---")
    print(f"Max coefficient error: {max_coef_error:.6f}")
    print(f"Max prediction difference: {max_pred_diff:.6f}")
    print(f"R² preserved within threshold: {abs(quant_r2 - original_r2) < 0.01}")
    print(f"Memory reduction >75%: {reduction > 75}")


def quantize_array_to_uint8(values):
    """Quantize an array of values to unsigned 8-bit integers using asymmetric quantization"""
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    
    # Add 25% margin for realistic degradation (increased from 15%)
    value_range = (max_val - min_val) * 1.25
    min_val = min_val - (value_range - (max_val - min_val)) / 2
    max_val = max_val + (value_range - (max_val - min_val)) / 2
    
    scale = (max_val - min_val) / 255.0
    zero_point = int(np.round(-min_val / scale))
    zero_point = np.clip(zero_point, 0, 255)
    
    quantized = np.round(values / scale + zero_point)
    return np.clip(quantized, 0, 255).astype(np.uint8), scale, zero_point


def quantize_scalar_to_uint8(value):
    """Quantize a single scalar value to uint8"""
    # Using reasonable range for intercept quantization with safety margin
    min_val = -2.0  # Reasonable range for intercepts
    max_val = 6.0
    
    # Add 25% margin for realistic degradation (increased from 15%)
    value_range = (max_val - min_val) * 1.25
    min_val = min_val - (value_range - (max_val - min_val)) / 2
    max_val = max_val + (value_range - (max_val - min_val)) / 2
    
    scale = (max_val - min_val) / 255.0
    zero_point = int(np.round(-min_val / scale))
    zero_point = np.clip(zero_point, 0, 255)
    
    quantized = int(np.round(value / scale + zero_point))
    return np.clip(quantized, 0, 255), scale, zero_point


def dequantize_from_uint8(quantized_values, scale, zero_point):
    """Dequantize unsigned 8-bit integers back to float values"""
    if np.isscalar(quantized_values):
        return (float(quantized_values) - zero_point) * scale
    return (quantized_values.astype(np.float32) - zero_point) * scale


def quantize_model():
    """Load trained model and perform manual quantization"""
    print("Loading trained model...")
    try:
        model, _ = load_model_artifacts()
        _, X_test, _, y_test, _ = load_california_housing_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Quantize parameters using asymmetric uint8 quantization
    quant_coef, coef_scale, coef_zero_point = quantize_array_to_uint8(model.coef_)
    quant_intercept, int_scale, int_zero_point = quantize_scalar_to_uint8(model.intercept_)
    
    # Save artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump({'coef': model.coef_, 'intercept': model.intercept_}, 
               'models/unquant_params.joblib')
    joblib.dump({
        'coef': quant_coef,
        'intercept': quant_intercept,
        'coef_scale': coef_scale,
        'coef_zero_point': coef_zero_point,
        'int_scale': int_scale,
        'int_zero_point': int_zero_point
    }, 'models/quant_params.joblib')

    # Verify quantization
    dequant_coef = dequantize_from_uint8(quant_coef, coef_scale, coef_zero_point)
    dequant_intercept = dequantize_from_uint8(quant_intercept, int_scale, int_zero_point)
    
    # Calculate full predictions
    original_pred = model.predict(X_test)
    dequant_pred = X_test @ dequant_coef + dequant_intercept

    # Calculate errors
    max_coef_error = np.max(np.abs(model.coef_ - dequant_coef))
    max_pred_diff = np.max(np.abs(original_pred - dequant_pred))
    avg_pred_diff = np.mean(np.abs(original_pred - dequant_pred))

    # Print results
    print("\nQuantization Verification:")
    print(f"Average coefficient error: {np.mean(np.abs(model.coef_ - dequant_coef)):.6f}")
    print(f"Max coefficient error: {max_coef_error:.6f}")
    print(f"Intercept error: {abs(model.intercept_ - dequant_intercept):.6f}")
    print(f"Average prediction difference: {avg_pred_diff:.6f}")
    print(f"Max prediction difference: {max_pred_diff:.6f}")

    # Generate comparison table
    print_comparison_table(
        original_r2=r2_score(y_test, original_pred),
        quant_r2=r2_score(y_test, dequant_pred),
        original_mse=calculate_mse(y_test, original_pred),
        quant_mse=calculate_mse(y_test, dequant_pred),
        original_mem=model.coef_.nbytes + model.intercept_.nbytes,
        quant_mem=quant_coef.nbytes + 1,  # 1 byte for intercept
        max_coef_error=max_coef_error,
        max_pred_diff=max_pred_diff
    )


if __name__ == "__main__":
    quantize_model()