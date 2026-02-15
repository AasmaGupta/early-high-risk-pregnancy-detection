from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# Load model once
model = joblib.load("maternal_risk_pipeline.pkl")

risk_map = {
    0: "Low Risk",
    1: "Mid Risk",
    2: "High Risk"
}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([{
        "Age": data["age"],
        "SystolicBP": data["sys_bp"],
        "DiastolicBP": data["dia_bp"],
        "BS": data["bs"],
        "BodyTemp": data["temp"],
        "HeartRate": data["hr"]
    }])

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]

    return {
        "risk": risk_map[pred],
        "confidence": float(max(proba))
    }

