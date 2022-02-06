# Script to train machine learning model.

# Add the necessary imports for the starter code.
from pathlib import Path
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from ml.model import Inference_artifact

FOLDER_DATA = Path("data")
FOLDER_MODEL = Path("model")
# Add code to load in the data.

data = pd.read_csv( FOLDER_DATA/"census_clean.csv", low_memory=False)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features, 
    label="salary", 
    training=False, 
    encoder=encoder, 
    lb=lb, 
)

# Train and save a model.
clf = train_model(X_train, y_train)
with open( FOLDER_MODEL/'model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# save encoder data
with open( FOLDER_MODEL/'onehot_encoder.pkl', 'wb') as f:
    pickle.dump((encoder, lb, cat_features), f)

# Load and test model 
inference_artifact = Inference_artifact( 
                                        FOLDER_MODEL/'model.pkl',
                                        FOLDER_MODEL/'onehot_encoder.pkl',
                                         )


# Performance in Slices
def slice_metrics(df, feature, file=None):
    """ 
        Function for calculating the performance of the model on slices of the data.
        Only for categorical features
    """
    print(f"------------ feature: {feature} ------------", file=file)
    for cls in df[feature].unique():
        df_temp = df[df[feature] == cls]

        _,y_test_temp,_,_ = inference_artifact.onehot_encoder.process_data(df_temp, 'salary')
        preds = inference_artifact.predict(df_temp)
        metrics = compute_model_metrics(y_test_temp, preds)

        print(f"feature: {feature}, value: {cls}", file=file)
        print(f"metrics: {metrics}", file=file)
    print(file=file)


preds = inference_artifact.predict(test)

SLICE_OUTPUT_FILE = Path("tests/slice_output.txt")
SLICE_OUTPUT_FILE.unlink(missing_ok=True) # deletes file if exists
with open(SLICE_OUTPUT_FILE, 'a') as f:
    print("Metrics: (precision, recall, fbeta)", file=f)
    print(f"General performance: {compute_model_metrics(y_test, preds)} \n", file=f)
    for feature in cat_features:
        slice_metrics(test, feature, f)