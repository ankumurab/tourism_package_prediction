# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi


# Load the dataset directly from the Hugging Face data space.
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/nishithworld/tourism-package-prediction/tourism.csv"
#df = pd.read_csv(DATASET_PATH)
df = pd.read_csv(
    DATASET_PATH,
    storage_options={
        "token": os.getenv("HF_TOKEN"),
        "timeout": 60
    }
)
print("Dataset loaded successfully.")

# Perform data cleaning and remove any unnecessary columns

# Drop Non-Predictive Columns
df.drop(columns=['CustomerID'], inplace=True)

# Target Variable
target_col = "ProdTaken"

# Split the cleaned dataset into training and testing sets, and save them locally
# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


#Upload the resulting train and test datasets back to the Hugging Face data space
files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="nishithworld/tourism-package-prediction",
        repo_type="dataset",
    )
