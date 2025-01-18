from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Logistic Regression Model API", description="API for Anomaly Detection")

model_path = "logistic_regression_model.pkl"
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    raise Exception("Model file not found. Please train and save the model as logistic_regression_model.pkl")

class PredictionInput(BaseModel):
    VIX: float
    DXY: float
    GTDEM2Y: float
    EONIA: float
    GTITL30YR: float
    GTITL2YR: float
    GTITL10YR: float
    GTJPY30YR: float
    GTJPY2YR: float

class PredictionOutput(BaseModel):
    is_anomaly: bool
    probability: float

@app.get("/")
def read_root():
    return {"message": "Logistic Regression Model API is running."}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    try:
        input_data = np.array([[data.VIX, data.DXY, data.GTDEM2Y, data.EONIA, data.GTITL30YR, 
                                data.GTITL2YR, data.GTITL10YR, data.GTJPY30YR, data.GTJPY2YR]])
        prob = model.predict_proba(input_data)[0][1] 
        is_anomaly = prob >= 0.5
        return PredictionOutput(is_anomaly=is_anomaly, probability=prob)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {e}")
