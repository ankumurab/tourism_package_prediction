# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-tourism-package-prediction")

api = HfApi()


#Load the train and test data from the Hugging Face data space
Xtrain_path = "hf://datasets/nishithworld/tourism-package-prediction/Xtrain.csv"
Xtest_path = "hf://datasets/nishithworld/tourism-package-prediction/Xtest.csv"
ytrain_path = "hf://datasets/nishithworld/tourism-package-prediction/ytrain.csv"
ytest_path = "hf://datasets/nishithworld/tourism-package-prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

##Performing the data cleaning

#Numerical Features (No scaling required for XGBoost)
numeric_features = [
    "Age",
    "MonthlyIncome",
    "DurationOfPitch",
    "NumberOfFollowups",
    "NumberOfTrips",
    "NumberOfPersonVisiting",
    "NumberOfChildrenVisiting"
]

#Binary Features
binary_features = [
    "Passport",
    "OwnCar"
]

#Ordinal Features (Keep numeric order)
ordinal_features = [
    "CityTier",                 # 1 > 2 > 3
    "PreferredPropertyStar",    # hotel rating
    "PitchSatisfactionScore"    # satisfaction score
]

# Categorical Features (One-Hot Encode)
categorical_features = [
    "Gender",
    "MaritalStatus",
    "Occupation",
    "Designation",
    "ProductPitched",
    "TypeofContact"
]


#Pipelines
#numeric
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

#categorical
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features + ordinal_features + binary_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

#Define a model and parameters  

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    #Tune the model with the defined parameters 
    # Hyperparameter tuning 
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    #Evaluating the model performance 

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_tourism_package_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "nishithworld/tourism_package_prediction"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_tourism_package_prediction_model_v1.joblib",
        path_in_repo="best_tourism_package_prediction_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
