# MLOps Pipeline for Linear Regression with Manual Quantization

This project implements a complete MLOps pipeline for the California Housing price prediction problem. The implementation covers training, testing, quantization, containerization, and automated deployment.

## Project Overview

The pipeline includes:

- Linear Regression model training on California housing data
- Comprehensive test suite using pytest
- Manual uint8 quantization for model compression
- Docker containerization for prediction testing
- Automated CI/CD pipeline with GitHub Actions

## Performance Summary

Managed to compress the model by **87.5%** while barely affecting performance:

| Measured           | Before   | After   | Change             |
| ------------------ | -------- | ------- | ------------------ |
| R² Score           | 0.5758   | 0.5746  | -0.0012            |
| Mean Squared Error | 0.5559   | 0.5574  | +0.0015            |
| Memory Usage       | 72 bytes | 9 bytes | **87.5% smaller!** |

The quantization introduces tiny errors (max coefficient error: 0.003277) but saves massive memory.

## How Everything is Organized

```
mlops-assignment/
├── .github/
│   └── workflows/
│       └── ci.yaml              # CI/CD pipeline configuration
├── src/
│   ├── __init__.py
│   ├── train.py                 # Model training script
│   ├── quantize.py              # Manual quantization implementation
│   ├── predict.py               # Prediction script for Docker verification
│   └── utils.py                 # Utility functions
├── tests/
│   ├── __init__.py
│   └── test_train.py            # Unit tests for training pipeline
├── models/                      # Generated model artifacts
│   ├── linear_regression.joblib # Trained model
│   ├── scaler.joblib           # Feature scaler
│   ├── unquant_params.joblib   # Original model parameters
│   └── quant_params.joblib     # Quantized model parameters
├── Dockerfile                   # Container configuration
├── requirements.txt             # Python dependencies
├── requirements.in              # Dependency source file
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Getting Started

### Prerequisites

- Python 3.13+
- Docker (for containerization)
- Git (for version control)

### Set Up Environment

1. Clone the repository:

```bash
git clone https://github.com/abhijeetk431/MLOps_Major.git
cd MLOps_Major
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate.ps1
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Model Training

Run the training script to train the Linear Regression model:

```bash
python src/train.py
```

Expected output:

```
Dataset Shapes:
Train features: (16512, 8), Test features: (4128, 8)
Train targets: (16512,), Test targets: (4128,)

Training Metrics:
Training R² Score: 0.6126
Training MSE (Loss): 0.5179

Test Metrics:
Test R² Score: 0.5758
Test MSE (Loss): 0.5559

Model Parameters for Quantization:
Coefficients: [ 0.85438303  0.12254624 -0.29441013  0.33925949 -0.00230772 -0.0408291
 -0.89692888 -0.86984178]
Intercept: 2.0719
```

The model achieves approximately 57% accuracy on housing price prediction.

### Testing

Run the test suite to validate the implementation:

```bash
python -m pytest tests/ -v
```

All 8 tests should pass, covering data loading, model training, performance validation, and model creation.

### Model Quantization

Execute the quantization script:

```bash
python src/quantize.py
```

The quantization process produces output showing:

```
Quantization Verification:
Average coefficient error: 0.001451
Max coefficient error: 0.003277
Intercept error: 0.014328
Average prediction difference: 0.014169
Max prediction difference: 0.304971

==================================================
       QUANTIZATION PERFORMANCE COMPARISON
==================================================

--- Model Performance ---
Metric          | Original     | Quantized    | Change
---------------------------------------------
R² Score        | 0.5758       | 0.5746       | -0.0012
MSE             | 0.5559       | 0.5574       | +0.0015

--- Memory Usage ---
Original params: 72.00 B
Quantized params: 9.00 B → 87.5% smaller

--- Verification ---
Max coefficient error: 0.003277
Max prediction difference: 0.304971
R² preserved within threshold: True
Memory reduction >75%: True
```

The quantization achieves 87.5% memory reduction with only 0.0012 loss in R² score.

### Model Prediction

Test the trained model:

```bash
python src/predict.py
```

## Docker Deployment

### Building the Container

Build the Docker image:

```bash
docker build -t mlops-linear-regression .
```

### Running the Container

Execute the containerized application:

```bash
docker run --rm mlops-linear-regression
```

Expected output:

```
Loading trained model...
Model loaded successfully!
Loading test data...
Test set size: (4128, 8)
Number of features: 8
Making predictions...

Prediction Statistics:
Number of predictions: 4128
Prediction range: [-1.0138, 11.5003]
Mean prediction: 2.0515
Std prediction: 0.9161

Sample Predictions (first 10):
Predicted       Actual          Difference
----------------------------------------
0.7191          0.4770          0.2421
1.7640          0.4580          1.3060
2.7097          5.0000          -2.2904
2.8389          2.1860          0.6529
2.6047          2.7800          -0.1753
2.0118          1.5870          0.4248
2.6455          1.9820          0.6635
2.1688          1.5750          0.5938
2.7407          3.4000          -0.6593
3.9156          4.4660          -0.5504

Mean Absolute Error on test set: 0.5332

Model Information:
Number of coefficients: 8
Intercept: 2.0719
Coefficient range: [-0.8969, 0.8544]

Prediction completed successfully!
```

## CI/CD Pipeline

Set up a GitHub Actions workflow that runs automatically when you push code. It does three things:

### 1. Test Everything First

Runs all the unit tests with pytest to make sure nothing's broken. If the tests fail, the pipeline stops here (which is good - we don't want to deploy broken code!).

### 2. Train and Quantize the Model

Once tests pass, it trains the Linear Regression model and runs the quantization. All the model files get saved as artifacts so the next job can use them.

### Pipeline Jobs

1. **Test Suite** - Validates code quality through comprehensive unit testing
2. **Train and Quantize** - Executes model training and quantization processes
3. **Build and Test Container** - Builds Docker image and validates containerized functionality

The pipeline triggers automatically on pushes to the main branch and pull requests.

## Testing Framework

The test suite provides comprehensive validation of the ML pipeline:

### Test Categories

- **Data Loading** - Validates California Housing dataset loading and preprocessing
- **Model Training** - Verifies LinearRegression initialization and training processes
- **Performance Validation** - Ensures R² score meets minimum threshold (≥0.5)
- **Artifact Management** - Tests model and scaler file creation and loading
- **Metric Calculation** - Validates accuracy of performance metrics

### Test Execution

```bash
# Run complete test suite
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_train.py::TestDataLoading -v
python -m pytest tests/test_train.py::TestModelPerformance -v
python -m pytest tests/test_train.py::TestArtifacts -v
```

All 8 tests should pass for successful validation.

## Technical Implementation

### Model Training

- **Dataset**: California Housing (20,640 samples, 8 features)
- **Algorithm**: Linear Regression (scikit-learn)
- **Preprocessing**: StandardScaler normalization
- **Data Split**: 80% training, 20% testing
- **Performance**: R² = 0.5758, MSE = 0.5559

### Manual Quantization

- **Method**: Asymmetric uint8 quantization (0-255 range)
- **Parameters**: Coefficients and intercept converted to 8-bit unsigned integers
- **Storage**: Scale and zero-point parameters for dequantization
- **Compression**: 87.5% memory reduction (72B → 9B)
- **Performance Impact**: Minimal degradation (-0.0012 R²)

### Containerization

- **Base Image**: `python:3.13-slim`
- **Dependencies**: Installed from requirements.txt
- **Entry Point**: Executes prediction script
- **Size**: Optimized for production deployment

## 🛠️ Development

### Adding Dependencies

## Technical Details

### Dataset Specifications

- **Source**: California Housing dataset from scikit-learn
- **Size**: 20,640 samples with 8 features
- **Target**: Median house value prediction
- **Data Split**: 80% training, 20% testing

### Model Performance

- **Algorithm**: Linear Regression
- **Training R²**: 0.6126
- **Test R²**: 0.5758 (approximately 57% accuracy)
- **Test MSE**: 0.5559

### Quantization Implementation

- **Method**: Manual uint8 asymmetric quantization
- **Memory Reduction**: 87.5% (from 72 bytes to 9 bytes)
- **Performance Impact**: -0.0012 R² score loss
- **Quantized R²**: 0.5746

### Dependencies

To add new packages:

1. Add the package to `requirements.in`
2. Regenerate requirements:
   ```bash
   pip-compile requirements.in
   ```
3. Install updated dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Generated Artifacts

Running `train.py` and `quantize.py` creates the following files in the `models/` directory:

- `linear_regression.joblib`: Trained LinearRegression model
- `scaler.joblib`: StandardScaler for feature normalization
- `unquant_params.joblib`: Original float64 model parameters
- `quant_params.joblib`: Quantized uint8 parameters with scaling metadata

## Assignment Requirements

This project fulfills all specified requirements:

- ✅ **Repository Setup**: README, .gitignore, and requirements.txt
- ✅ **Model Training**: LinearRegression on California Housing dataset
- ✅ **Testing Framework**: Comprehensive unit tests for all components
- ✅ **Manual Quantization**: uint8 quantization implementation with memory optimization
- ✅ **Docker Containerization**: Container with prediction script execution
- ✅ **CI/CD Pipeline**: 3-job automated workflow with proper dependencies
- ✅ **Performance Comparison**: Quantization metrics analysis and documentation

## Results Summary

| Component          | Status | Details                                       |
| ------------------ | ------ | --------------------------------------------- |
| **Model Training** | ✅     | R² = 0.5758, MSE = 0.5559                     |
| **Quantization**   | ✅     | R² = 0.5746, MSE = 0.5574                     |
| **Unit Tests**     | ✅     | 8/8 tests passing                             |
| **Quantization**   | ✅     | 87.5% memory reduction, minimal accuracy loss |
| **Docker**         | ✅     | Container execution validated                 |
| **CI/CD**          | ✅     | All pipeline jobs operational                 |

---

**Author**: ABHIJEET KUMAR  
**Course**: MLOps  
**Institution**: IIT Jodhpur  
**Date**: August 2, 2025
