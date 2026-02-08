import streamlit as st
import pandas as pd
import joblib

st.title("👩‍⚕️ Doctor Analytics Interface")

model = joblib.load("maternal_risk_pipeline.pkl")

age = st.number_input("Age")
sbp = st.number_input("Systolic BP")
dbp = st.number_input("Diastolic BP")
bs = st.number_input("Blood Sugar")
temp = st.number_input("Body Temp")
hr = st.number_input("Heart Rate")

if st.button("Run Prediction"):
    input_df = pd.DataFrame([{
        "Age": age,
        "SystolicBP": sbp,
        "DiastolicBP": dbp,
        "BS": bs,
        "BodyTemp": temp,
        "HeartRate": hr
    }])

    pred = model.predict(input_df)
    proba = model.predict_proba(input_df)

    st.write("Prediction:", pred)
    st.write("Probabilities:", proba)
    st.dataframe(input_df)
