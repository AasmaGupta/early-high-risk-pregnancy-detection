import streamlit as st

st.title("⚙️ Tech Stack")

cols = st.columns(3)
tech = ["Python","Scikit-learn","XGBoost","SHAP","Pandas","NumPy","Matplotlib","Streamlit","Joblib","GitHub"]

for i,t in enumerate(tech):
    cols[i%3].success(t)
