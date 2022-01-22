# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.

from pathlib import Path
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import pandas as pd
import pickle

FOLDER_DATA = Path("data")
FOLDER_MODEL = Path("model")
# Add code to load in the data.

data = pd.read_csv( FOLDER_DATA/"census_clean.csv", low_memory=False)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# Load and test model 

with open( FOLDER_MODEL/'model.pkl', 'rb') as f:
    clf = pickle.load(f)

preds = inference(clf, X_test)
print(f"General performance: {compute_model_metrics(y_test, preds)} \n")


# Performance in Slices

def slice_metrics(df, feature):
    """ 
        Function for calculating the performance of the model on slices of the data.
        Only for categorical features
    """
    print(f"------------ feature: {feature} ------------")
    for cls in df[feature].unique():
        df_temp = df[df[feature] == cls]

        X_test_temp, y_test_temp, _, _ = process_data(
            df_temp,
            categorical_features=cat_features, 
            label="salary", 
            training=False, 
            encoder=encoder, 
            lb=lb, 
            )

        preds = inference(clf, X_test_temp)
        metrics = compute_model_metrics(y_test_temp, preds)

        print(f"feature: {feature}, value: {cls}")
        print(f"metrics: {metrics}")
    print()

for feature in cat_features:
    slice_metrics(test, feature)