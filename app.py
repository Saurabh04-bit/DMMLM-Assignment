import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset to get feature names
data = pd.read_csv("winequality-red.csv")
feature_names = list(data.columns[:-1])  # exclude target "quality"

st.title("🍷 Wine Quality Prediction App")
st.write("Enter the wine's chemical properties to predict its quality score.")

# Create input form
input_data = {}
with st.form("wine_form"):
    for feat in feature_names:
        # Use number_input for continuous values
        val = st.number_input(f"{feat}", 
                              value=float(data[feat].mean()), 
                              step=0.01)
        input_data[feat] = val

    submitted = st.form_submit_button("Predict Quality")

# Prediction
if submitted:
    features = [input_data[feat] for feat in feature_names]
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)[0]

    st.success(f"Predicted Wine Quality: **{int(prediction)}**")
    st.write("Input Values:", input_data)


