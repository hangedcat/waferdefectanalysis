from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
import polars as pl
import joblib

app = FastAPI()
pipeline: Pipeline = joblib.load('finalpipe.joblib')

class Input(BaseModel):
    features: list[float] = Field(min_length=2, max_length=2)

@app.post('/predict')
def predict(data):
    df = pl.DataFrame([data.features])
    prediction = pipeline.predict(df)
    return {'prediction': prediction.tolist()} #type: ignore