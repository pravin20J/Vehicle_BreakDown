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

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/pravin1214/vehicle_break_down/engine_data.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.");

# Dropping Duplicates
df.drop_duplicates(inplace=True);

target_col = 'Engine Condition'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Scaling the X dataset to snadrad since each one has different units
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(Xtrain),
    columns=Xtrain.columns
)

X_test_scaled = pd.DataFrame(
    scaler.transform(Xtest),
    columns=Xtest.columns
)


X_train_scaled.to_csv("engine-maintenance-dataset/data/Xtrain.csv",index=False)
X_test_scaled.to_csv("engine-maintenance-dataset/data/Xtest.csv",index=False)
ytrain.to_csv("engine-maintenance-dataset/data/ytrain.csv",index=False)
ytest.to_csv("engine-maintenance-dataset/data/ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"];

for file_path in files:

    api.upload_file(
        path_or_fileobj="engine-maintenance-dataset/data/"+file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="pravin1214/vehicle_break_down",
        repo_type="dataset",
    )
