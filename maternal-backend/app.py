from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML model once
model = joblib.load("model/maternal_risk_pipeline.pkl")


class MaternalInput(BaseModel):
    age: float
    systolic_bp: float
    diastolic_bp: float
    blood_sugar: float
    body_temp: float
    heart_rate: float


@app.get("/")
def home():
    return {"message": "Maternal Risk API Running ✅"}


@app.post("/predict")
def predict(data: MaternalInput):

    features = np.array([[
        data.age,
        data.systolic_bp,
        data.diastolic_bp,
        data.blood_sugar,
        data.body_temp,
        data.heart_rate
    ]])

    prediction = model.predict(features)[0]

    return {
        "risk_level": str(prediction)
    }