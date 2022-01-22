from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import pickle
import pandas as pd
from .data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    clf = RandomForestClassifier(n_estimators=75, max_depth=50, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

class Inference_artifact:
    def __init__(self, model_path: Path):
        with open( model_path, 'rb') as f:
            (self.clf, self.encoder, self.lb, self.cat_features) = pickle.load(f)
    
    def predict(self, data: pd.DataFrame):
        if 'salary' in data:
            data = data.copy()
            data.pop('salary')
        X, _, _, _ = process_data(
            data,
            categorical_features=self.cat_features, 
            label=None, 
            training=False, 
            encoder=self.encoder, 
            lb=self.lb, 
        )
        pred = inference(self.clf, X)
        return pred