from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/LightGBM.pkl")

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def predict(transaction: list):

    data = np.array(transaction).reshape(1, -1)
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "fraud_prediction": int(prediction),
        "risk_score": float(probability)
    }