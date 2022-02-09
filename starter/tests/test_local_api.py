import pytest
import requests
import pandas as pd
import pickle
from pathlib import Path
from fastapi.testclient import TestClient
from main import app

from starter.ml.data import process_data
from starter.ml.model import inference

home = TestClient(app)

@pytest.fixture(scope="session")
def data():
    FOLDER_DATA = Path("data")
    data = pd.read_csv( FOLDER_DATA/"census_clean.csv", low_memory=False)
    return data

@pytest.fixture(scope="session")
def model():
    FOLDER_MODEL = Path("model")
    with open( FOLDER_MODEL/'model.pkl', 'rb') as f:
        clf = pickle.load(f)
    return clf

@pytest.fixture(scope="session")
def encoder_data():
    FOLDER_MODEL = Path("model")
    with open( FOLDER_MODEL/'onehot_encoder.pkl', 'rb') as f:
        (encoder, lb, cat_features) = pickle.load(f)
    return (encoder, lb, cat_features)

@pytest.fixture(scope="session")
def processed_data(data, encoder_data):
    """
        returns (X, y, encoder, lb)
    """
    (encoder, lb, cat_features) = encoder_data
    (X, y, encoder, lb) = process_data(
        data,
        categorical_features=cat_features, 
        label="salary", 
        training=False,
        encoder = encoder,
        lb = lb
        )
    return (X, y, encoder, lb)

def test_say_hello():
    r = home.get("/")
    assert r.status_code == 200
    assert "greetings" in r.json()

def test_predict_positive(data, processed_data, model):
    (X, _, _, _) = processed_data
    clf = model    
    # Get 10 rows with positive label
    Xi = X[data['salary']=='>50K'][:10]
    # Model prediction for 10 values
    preds = inference(clf, Xi).tolist()
    #Find index of positive prediction 
    # note: (not all rows with positive label will get a positive prediction)
    idx = preds.index(1)
    # Select index row from dataframe
    data_json = data[data['salary']=='>50K'].head(10).iloc[idx].to_json()
    # Api prediction
    r = home.post("/predict", data_json)
    assert r.status_code == 200
    assert r.json()['prediction']==[1]

def test_predict_negative(data, processed_data, model, encoder_data):
    (X, _, _, _) = processed_data
    clf = model    
    # Get 10 rows with negative label
    Xi = X[data['salary']=='<=50K'][:10]
    # Model prediction for 10 values
    preds = inference(clf, Xi).tolist()
    #Find index of negative prediction 
    # note: (not all rows with negative label will get a negative prediction)
    idx = preds.index(0)
    # Select index row from dataframe
    data_json = data[data['salary']=='<=50K'].head(10).iloc[idx].to_json()
    # Api prediction
    r = home.post("/predict", data_json)
    assert r.status_code == 200
    assert r.json()['prediction']==[0]

def test_predict_alldata(data, processed_data, model, encoder_data):
    (X, y, _, _) = processed_data
    clf = model
    (encoder, lb, cat_features) = encoder_data
    preds = inference(clf, X).tolist()

    data_json = data.to_json()
    r = home.post("/predict", data_json)
    preds2 = r.json()['prediction']
    assert r.status_code == 200
    assert preds.__str__() == preds2.__str__()