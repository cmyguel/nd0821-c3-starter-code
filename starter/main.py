# Put the code for your API here.
import pandas as pd
from fastapi import FastAPI
from typing import Union, Dict
from pydantic import BaseModel, Field
from pathlib import Path

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
    age:            Union[int, Dict[str, int]]
    workclass:      Union[str, Dict[str, str]]
    fnlgt:          Union[int, Dict[str, int]]
    education:      Union[str, Dict[str, str]]
    occupation:     Union[str, Dict[str, str]]
    relationship:   Union[str, Dict[str, str]]
    race:           Union[str, Dict[str, str]]
    sex:            Union[str, Dict[str, str]]
    salary:         Union[str, Dict[str, str]]
    education_num:  Union[int, Dict[str, int]] = Field(alias='education-num')
    marital_status: Union[str, Dict[str, str]] = Field(alias='marital-status')
    capital_gain:   Union[int, Dict[str, int]] = Field(alias='capital-gain')
    capital_loss:   Union[int, Dict[str, int]] = Field(alias='capital-loss')
    hours_per_week: Union[int, Dict[str, int]] = Field(alias='hours-per-week')
    native_country: Union[str, Dict[str, str]] = Field(alias='native-country')

@app.post('/predict')
async def predict(request: RequestModel):
    
    input_data = request.dict()

    # change underlines in keys from input_data keys 
    # (_) -> (-)
    for k in list(input_data.keys()):
        if "_" in k:
            input_data[k.replace('_', '-')] = input_data.pop(k)

    input_df = pd.DataFrame(input_data)
    if 'salary' in input_df:
        input_df.pop('salary')
    
    pred = inference.predict(input_df)
    return {"prediction": pred.tolist()}