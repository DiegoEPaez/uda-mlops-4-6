# Put the code for your API here.
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from .ml.data import process_data
from .ml.model import inference, load_model


class Census(BaseModel):
    age: int = Field(..., alias="age")
    workclass: str = Field(..., alias="workclass")
    education: str = Field(..., alias="education")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str = Field(..., alias="occupation")
    relationship: str = Field(..., alias="relationship")
    race: str = Field(..., alias="race")
    sex: str = Field(..., alias="sex")
    capital_gain: float = Field(..., alias="capital-gain")
    capital_loss: float = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "education": "Bachelors",
                    "marital-status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 5000,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States",
                }
            ]
        },
    )


app = FastAPI()


@app.get("/")
async def welcome():
    return "Welcome to Income Predictor"


@app.post("/inference")
async def inference_ep(census: Census):
    df = pd.DataFrame([census.model_dump(by_alias=True)])
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

    model, encode, lb = load_model()
    X, _, _, _ = process_data(
        df, cat_features, label=None, training=False, encoder=encode, lb=lb
    )
    preds = inference(model, X)
    out = preds.tolist()

    return {"predictions": out}
