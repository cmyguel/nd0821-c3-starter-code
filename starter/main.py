# Put the code for your API here.
import pandas as pd
import pickle
from fastapi import FastAPI, Request
from typing import Union, List
from pydantic import BaseModel, Field
from pathlib import Path

from starter.ml.data import process_data
from starter.ml.model import Inference_artifact



# instantiate the app
app = FastAPI()
# instantiate inference artifact
MODEL_PATH = Path("model/model_encoder_lb_catfeatures.pkl")
inference = Inference_artifact(MODEL_PATH) 

# GET welcome message
@app.get('/')
async def say_hello():
    return {"greetings": "Welcome to the >50k maker detection API."}

class RequestModel(BaseModel):
    age: dict
    workclass: dict
    fnlgt: dict
    education: dict
    education_num: dict = Field(alias='education-num')
    marital_status: dict = Field(alias='marital-status')
    occupation: dict
    relationship: dict
    race: dict
    sex: dict
    capital_gain: dict = Field(alias='capital-gain')
    capital_loss: dict = Field(alias='capital-loss')
    hours_per_week: dict = Field(alias='hours-per-week')
    native_country: dict = Field(alias='native-country')
    salary: dict

@app.post('/')
async def predict(request: RequestModel):
    # input_data = await request.json()
    input_data = request.dict()

    # change underlines by dashes in keys from input_data keys 
    # (_) -> (-)
    for k in list(input_data.keys()):
        if "_" in k:
            input_data[k.replace('_', '-')] = input_data.pop(k)

    print(50*"-")
    print("input_data.keys():")
    print(input_data.keys())

    print(50*"-")
    print("types:")
    for key,val in input_data.items():
        print(key, type(val), type([v for v in val.values()][0]))

    input_df = pd.DataFrame(input_data)
    if 'salary' in input_df:
        input_df.pop('salary')
    
    pred = inference.predict(input_df)
    return {"prediction": pred.tolist()}


# @app.post('/')
# async def predict(request: Request):
#     input_data = await request.json()

#     input_df = pd.DataFrame(input_data)
#     if 'salary' in input_df:
#         input_df.pop('salary')
    
#     pred = inference.predict(input_df)

#     return {"prediction": pred.tolist()}