import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.title("ML Classification Model Predictor")

# Load scaler
scaler = joblib.load("models/scaler.pkl")

# Model selection
model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# Load model
model = joblib.load(f"models/{model_name}.pkl")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Data:")
    st.write(data.head())

    # Scale if needed
    if model_name in ["Logistic Regression", "KNN"]:
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)
    else:
        prediction = model.predict(data)

    st.write("Predictions:")
    st.write(prediction)
