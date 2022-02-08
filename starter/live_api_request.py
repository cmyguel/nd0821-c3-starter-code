import requests
import pandas as pd
from pathlib import Path
from starter.ml.data import process_data



FOLDER_DATA = Path("data")
URL = "https://udacity-50k-prediction.herokuapp.com"

data = pd.read_csv( FOLDER_DATA/"census_clean.csv", low_memory=False)

r = requests.get(URL)
print(r.json())

data_json = data.head(10).to_json()
r = requests.post( URL+'/predict', data_json)
print(r.json())