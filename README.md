This repository implements an end-to-end machine learning pipeline for predicting whether a customer will purchase a tourism package.
It follows MLOps best practices, including data preparation, model training, experiment tracking, containerization, and deployment readiness.

**Project Objective**

To build a binary classification model that predicts whether a customer will purchase a tourism package based on demographic, behavioral, and interaction features.

- Target Variable: ProdTaken (0 = No, 1 = Yes)
- Model: XGBoost Classifier
- Tracking: MLflow
- Deployment: Dockerized inference service
- Data Versioning: Hugging Face Datasets

**Machine Learning Pipeline**
1. Data Preparation (prep.py)
- Missing value imputation
- Feature categorization:
  - Numerical
  - Binary
  - Ordinal
  - Categorical (One-Hot Encoded)
- Train–test split
- Saves processed datasets for training

**Model Training (train.py)**
- Preprocessing using ColumnTransformer
- XGBoost classifier with class imbalance handling
- Hyperparameter tuning using GridSearchCV
- MLflow experiment tracking:
  - Parameters
  - Metrics
  - Best model artifacts
- Uploads trained model to **Hugging Face Model Hub**

**Experiment Tracking (MLflow)**
- Tracks:
  - Accuracy, Precision, Recall, F1-Score
  - Best hyperparameters
  - Model artifacts
 
**Deployment (deployment/)**
- app.py exposes the trained model for inference
- Dockerfile builds a containerized API
- Separate requirements.txt for lightweight serving

**CI/CD (.github/workflows/)**
- Automated pipeline:
  - Dependency installation
  - Data preparation
  - Model training
- Ensures reproducible builds
