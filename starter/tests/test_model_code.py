import pytest
import numpy as np
import pandas as pd
import sklearn as sk
from pathlib import Path
from ..starter.ml.data import process_data
from ..starter.ml.model import train_model, inference, compute_model_metrics

@pytest.fixture(scope="session")
def data():
    FOLDER_DATA = Path("data")
    data = pd.read_csv( FOLDER_DATA/"census_clean.csv", low_memory=False)
    return data

@pytest.fixture(scope="session")
def cat_features():
    return [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

@pytest.fixture(scope="session")
def processed_data(data, cat_features):
    """
        returns (X, y, encoder, lb)
    """
    (X, y, encoder, lb) = process_data(
        data,
        categorical_features=cat_features, 
        label="salary", 
        training=True,
        )
    return (X, y, encoder, lb)

@pytest.fixture(scope="session")
def clf(processed_data):
    (X, y, _, _) = processed_data
    return train_model(X, y)

def test_data_load(data, processed_data):
    (X, y, _, _) = processed_data
    assert isinstance(data, pd.DataFrame)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]

def test_train_model(clf):
    assert isinstance(clf, sk.ensemble._forest.RandomForestClassifier)

def test_inference(clf, processed_data):
    (X, y, _, _) = processed_data
    pred = inference(clf, X)
    assert X.shape[0] == pred.shape[0]
    assert y.shape.__str__() == pred.shape.__str__()

def test_metrics(clf, processed_data):
    (X, y, _, _) = processed_data
    preds = inference(clf, X)
    (precision, recall, fbeta) = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    # Assert model can overfit training data
    assert precision >= 0.8
    assert recall >= 0.8
    assert fbeta >= 0.8