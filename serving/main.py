from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
import polars as pl
import joblib
import sqlite3
import os

API_KEY = os.environ["MY_API_KEY"]
app = FastAPI()
pipeline: Pipeline = joblib.load('finalpipe.joblib')

conn = sqlite3.connect('history.db')
cur = conn.cursor()
cur.execute('''
    CREATE TABLE IF NOT EXISTS predictions(
        temperature REAL,
        pressure REAL,
        prediction INTEGER,
        timestamp TEXT DEFAULT TIMESTAMP
        )
''')

async def verify_api_key(x_api_key: str = Header()):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail='Invalid api key')

@app.get('/history')
def get_history():
    hist=[]
    for row in cur.execute("""SELECT * from predictions"""):
        hist.append(row)
    return hist

class Input(BaseModel):
    features: list[float] = Field(min_length=2, max_length=2)

@app.post('/predict', dependencies=[Depends(verify_api_key)])
def predict(data: Input):
    df = pl.DataFrame({'temperature' : data.features[0], 'pressure' : data.features[1]})
    prediction = pipeline.predict(df)

    cur.execute("INSERT INTO predictions VALUES(?, ?, ?)", (data.features[0], data.features[1], prediction[0]))   
    conn.commit()

    return {'prediction': prediction.tolist()} #type: ignore