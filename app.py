import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="ML Classification App",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning Classification Predictor")

st.write(
"""
Upload your test CSV file, select a trained model, and get predictions.
"""
)

# ===============================
# LOAD PREPROCESSING OBJECTS
# ===============================

try:
    scaler = joblib.load("models/scaler.pkl")
    column_transformer = joblib.load("models/column_transformer.pkl")
except:
    st.error("Preprocessing files not found. Make sure models folder is uploaded.")
    st.stop()


# ===============================
# MODEL FILE MAP
# ===============================

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

# ===============================
# MODEL SELECTION
# ===============================

model_name = st.selectbox(
    "Select Model",
    list(model_files.keys())
)

# Load selected model
try:
    model = joblib.load(f"models/{model_files[model_name]}")
except:
    st.error(f"{model_name} model file not found.")
    st.stop()


# ===============================
# FILE UPLOAD
# ===============================

uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

# ===============================
# PREDICTION
# ===============================

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    try:
        # Apply column transformer
        data_encoded = column_transformer.transform(data)

        # Apply scaling
        data_scaled = scaler.transform(data_encoded)

        # Convert sparse to dense if needed
        if hasattr(data_scaled, "toarray"):
            data_scaled = data_scaled.toarray()

        # For tree models use encoded (not scaled)
        if model_name in ["Decision Tree", "Random Forest", "XGBoost"]:
            prediction = model.predict(data_encoded)
        else:
            prediction = model.predict(data_scaled)

        # Show predictions
        st.subheader("Predictions")

        result_df = data.copy()
        result_df["Prediction"] = prediction

        st.dataframe(result_df)

        # Download option
        csv = result_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "Download Predictions",
            csv,
            "predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ===============================
# SHOW MODEL PERFORMANCE TABLE
# ===============================

st.subheader("Model Performance Comparison")

try:
    results = pd.read_csv("models/model_results.csv")
    st.dataframe(results)
except:
    st.warning("model_results.csv not found.")
