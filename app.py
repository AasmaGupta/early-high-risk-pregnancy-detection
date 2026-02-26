import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Hybrid Maternal Risk AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("maternal_risk_pipeline.pkl")

model = load_model()

# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Patient Interface",
        "Doctor Interface",
        "Model Insights",
        "System Architecture",
        "Tech Stack"
    ]
)

# ---------------------------------------------------
# SHARED INPUT FUNCTION
# ---------------------------------------------------
def patient_inputs():
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 16, 50, 25)
        sys = st.number_input("Systolic BP", 80, 200, 120)
        bs = st.number_input("Blood Sugar", 50, 300, 100)

    with col2:
        dia = st.number_input("Diastolic BP", 40, 150, 80)
        temp = st.number_input("Body Temperature", 95.0, 105.0, 98.6)
        hr = st.number_input("Heart Rate", 40, 180, 75)

    df = pd.DataFrame([{
        "Age": age,
        "SystolicBP": sys,
        "DiastolicBP": dia,
        "BS": bs,
        "BodyTemp": temp,
        "HeartRate": hr
    }])

    return df

# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------
if page == "Home":

    st.title("Hybrid Clinical–AI Maternal Risk Detection")

    st.markdown("""
    This system predicts pregnancy risk levels using a hybrid ensemble of:
    - Logistic Regression  
    - Random Forest  
    - XGBoost  
    - Soft Voting Ensemble  

    It integrates explainable AI to support clinical decision making.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.info("Navigate to Patient Interface for quick screening.")

    with col2:
        st.info("Doctors can explore advanced analytics and model insights.")

# ---------------------------------------------------
# PATIENT PAGE
# ---------------------------------------------------
elif page == "Patient Interface":

    st.header("Patient Risk Assessment")

    input_df = patient_inputs()

    if st.button("Predict Risk"):

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        risk_map = {0:"Low Risk",1:"Mid Risk",2:"High Risk"}
        result = risk_map[prediction]
        confidence = round(np.max(proba)*100,2)

        if result == "High Risk":
            st.error(f"Risk Level: {result}")
        elif result == "Mid Risk":
            st.warning(f"Risk Level: {result}")
        else:
            st.success(f"Risk Level: {result}")

        st.write(f"Confidence: {confidence}%")

        if result == "High Risk":
            st.info("Recommendation: Seek medical consultation immediately.")
        elif result == "Mid Risk":
            st.info("Recommendation: Monitor regularly.")
        else:
            st.info("Recommendation: Maintain routine prenatal care.")

# ---------------------------------------------------
# DOCTOR PAGE
# ---------------------------------------------------
elif page == "Doctor Interface":

    st.header("Clinical Decision Support Panel")

    input_df = patient_inputs()

    st.subheader("Input Summary")
    st.dataframe(input_df)

    if st.button("Run Clinical Prediction"):

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        risk_map = {0:"Low Risk",1:"Mid Risk",2:"High Risk"}
        result = risk_map[prediction]

        st.subheader("Prediction Result")
        st.write(result)

        st.subheader("Probability Breakdown")
        st.bar_chart(pd.DataFrame({
            "Risk Level": ["Low","Mid","High"],
            "Probability": proba
        }).set_index("Risk Level"))

# ---------------------------------------------------
# MODEL INSIGHTS
# ---------------------------------------------------
elif page == "Model Insights":

    st.header("Model Performance & Explainability")

    st.subheader("Model Accuracy Comparison (Example)")
    fig, ax = plt.subplots()
    models = ["LR","RF","XGB","Stack","Vote"]
    scores = [0.78,0.86,0.88,0.89,0.90]
    ax.bar(models, scores)
    st.pyplot(fig)

    st.subheader("Feature Importance Placeholder")
    st.info("Insert saved SHAP or Permutation plots here")

    st.subheader("Explainability")
    st.markdown("""
    This system supports:
    - SHAP value interpretation
    - Permutation importance
    - Partial dependence visualization
    """)

# ---------------------------------------------------
# ARCHITECTURE
# ---------------------------------------------------
elif page == "System Architecture":

    st.header("System Pipeline")

    st.markdown("""
    ```
    User Input
        ↓
    Data Preprocessing
        ↓
    Ensemble ML Models
        ↓
    Explainability Layer
        ↓
    Risk Classification Output
    ```
    """)

# ---------------------------------------------------
# TECH STACK
# ---------------------------------------------------
elif page == "Tech Stack":

    st.header("Technology Used")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("Python")
        st.success("Scikit-learn")
        st.success("XGBoost")

    with col2:
        st.success("SHAP")
        st.success("Pandas")
        st.success("NumPy")

    with col3:
        st.success("Streamlit")
        st.success("Matplotlib")
        st.success("GitHub")



