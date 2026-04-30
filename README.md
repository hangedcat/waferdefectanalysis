# Wafer Defect Analysis Pipeline

This project is a compact wafer defect classification demo built around a synthetic dataset, an exploratory notebook, a trained `scikit-learn` pipeline, and a FastAPI inference service.

## Overview

- `wafer_data_generator.py` creates the synthetic wafer dataset and writes `df.csv`.
- `EDA.ipynb` contains the exploratory analysis and feature inspection.
- `serving/main.py` loads `serving/finalpipe.joblib` and exposes a `/predict` endpoint.
- The saved pipeline accepts a two-value feature vector and returns a binary prediction.

## Project Structure

- `wafer_data_generator.py` - synthetic data generator
- `df.csv` - generated dataset
- `EDA.ipynb` - exploratory notebook
- `serving/main.py` - FastAPI serving app
- `serving/finalpipe.joblib` - serialized inference pipeline

## Data Generation

Run the generator from the project root:

```powershell
python wafer_data_generator.py
```

This recreates `df.csv` with 500 synthetic wafer records and a binary `passed` label derived from defect count.

## Serving the Model

Start the API from the `serving` directory:

```powershell
cd serving
uvicorn main:app --reload
```

`main.py` loads `finalpipe.joblib` at import time, so launching the app from `serving/` avoids path issues.

### Predict

`POST /predict`

Example request body:

```json
{
  "features": [12.4, 2.1]
}
```

Example response:

```json
{
  "prediction": [0]
}
```

The endpoint is designed for exactly two numeric feature values.

## Dependencies

The project uses:

- `numpy`
- `pandas`
- `mlflow`
- `scikit-learn`
- `fastapi`
- `pydantic`
- `polars`
- `joblib`
- `uvicorn`

Install them in your environment with `pip` or your preferred package manager.

## Notes

- `df.csv` is generated output.
- `serving/finalpipe.joblib` must exist before starting the API.
- `EDA.ipynb` is the best place to inspect the exploratory workflow and feature development trail.
