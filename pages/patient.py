import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

st.title("🤰 Patient Risk Check")

# -------- LOAD MODEL --------
model = joblib.load("maternal_risk_pipeline.pkl")

# -------- INPUT FORM --------
age = st.number_input("Age", 18, 50)
sbp = st.number_input("Systolic BP")
dbp = st.number_input("Diastolic BP")
bs = st.number_input("Blood Sugar")
temp = st.number_input("Body Temperature")
hr = st.number_input("Heart Rate")

if st.button("Predict Risk"):
    input_df = pd.DataFrame([{
        "Age": age,
        "SystolicBP": sbp,
        "DiastolicBP": dbp,
        "BS": bs,
        "BodyTemp": temp,
        "HeartRate": hr
    }])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    labels = ["Low","Mid","High"]
    risk = labels[pred]
    confidence = np.max(proba)*100

    color = {"Low":"green","Mid":"orange","High":"red"}[risk]

    st.markdown(f"""
    <div style='background:{color};padding:20px;border-radius:10px;text-align:center;color:white;font-size:24px'>
    Risk Level: {risk}<br>Confidence: {confidence:.2f}%
    </div>
    """, unsafe_allow_html=True)

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "Confidence"},
        gauge={'axis': {'range': [0,100]}}
    ))
    st.plotly_chart(fig)

    # Recommendation
    rec = {
        "High":"Immediate consultation advised",
        "Mid":"Monitoring recommended",
        "Low":"Routine care"
    }[risk]

    st.info(rec)

    with st.expander("Explanation"):
        st.write("This AI system analyzes health indicators to estimate pregnancy risk levels. It is not a medical diagnosis.")
