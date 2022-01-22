# Put the code for your API here.
import pandas as pd
from fastapi import FastAPI, Request
from typing import Union, List
from pydantic import BaseModel
from pathlib import Path
from starter.ml.data import process_data
from starter.ml.model import Inference_artifact

import pickle

# instantiate the app
app = FastAPI()
# instantiate inference artifact
MODEL_PATH = Path("model/model_encoder_lb_catfeatures.pkl")
inference = Inference_artifact(MODEL_PATH) 

# GET welcome message
@app.get('/')
async def say_hello():
    return {"greetings": "Welcome to the >50k maker detection API."}

@app.post('/')
async def predict(request: Request):
    input_data = await request.json()

    input_df = pd.DataFrame(input_data)
    if 'salary' in input_df:
        input_df.pop('salary')
    
    pred = inference.predict(input_df)

    return {"prediction": pred.tolist()}