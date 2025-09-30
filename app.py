import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys, sklearn

# --- Environment info ---
st.sidebar.title("Environment Info")
st.sidebar.write("**Python version:**", sys.version)
st.sidebar.write("**scikit-learn version:**", sklearn.__version__)
st.sidebar.write("**NumPy version:**", np.__version__)
st.sidebar.write("**Pandas version:**", pd.__version__)

# --- Load trained Random Forest model ---
model = joblib.load("random_forest_model.pkl")

# --- Load dataset to get feature names ---
data = pd.read_csv("winequality-red.csv")
feature_names = list(data.columns[:-1])  # all except "quality"

st.title("🍷 Wine Quality Prediction App")
st.write("Enter the wine's chemical properties to predict its quality score.")

# --- Input form ---
input_data = {}
with st.form("wine_form"):
    for feat in feature_names:
        val = st.number_input(
            f"{feat}",
            value=float(data[feat].mean()),
            step=0.01
        )
        input_data[feat] = val

    submitted = st.form_submit_button("Predict Quality")

# --- Prediction ---
if submitted:
    features = [input_data[feat] for feat in feature_names]
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)[0]

    st.success(f"Predicted Wine Quality: **{int(prediction)}**")
    st.write("🔎 Input Values:", input_data)



