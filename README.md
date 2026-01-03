This repository implements an end-to-end machine learning pipeline for predicting whether a customer will purchase a tourism package.
It follows MLOps best practices, including data preparation, model training, experiment tracking, containerization, and deployment readiness.

tourism_project/
│
├── .github/
│   └── workflows/
│       └── pipeline.yml        # CI pipeline (build, test, train)
│
├── tourism_project/
│   ├── deployment/
│   │   ├── Dockerfile          # Docker image for model serving
│   │   ├── app.py              # Inference application (API)
│   │   └── requirements.txt    # Deployment dependencies
│   │
│   ├── hosting/
│   │   └── hosting.py          # Hosting / serving logic
│   │
│   ├── model_building/
│   │   ├── data/
│   │   │   └── (datasets)      # Training & test datasets
│   │   │
│   │   ├── data_register.py    # Dataset registration (Hugging Face)
│   │   ├── prep.py             # Data preprocessing pipeline
│   │   ├── train.py            # Model training + MLflow logging
│   │   └── requirements.txt    # Training dependencies
│
├── README.md
