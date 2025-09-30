import streamlit as st
import pandas as pd
import joblib   # use joblib instead of pickle

# Load the trained Random Forest model
model_filename = "random_forest_model.pkl"
model = joblib.load(model_filename)

# Define feature columns (based on winequality-red dataset)
feature_columns = [
    'fixed acidity', 'volatile acidity', 'citric acid',
    'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH',
    'sulphates', 'alcohol'
]

# Prediction function
def predict_wine_quality(features):
    prediction = model.predict(features)
    return prediction

# Streamlit App UI
st.title("🍷 Wine Quality Prediction")
st.write("Enter the wine chemical properties to predict its quality (score 0-10).")

# Collect user input
input_data = []
for col in feature_columns:
    value = st.number_input(f"Enter {col}", min_value=0.0, step=0.1)
    input_data.append(value)

# Convert to dataframe
input_df = pd.DataFrame([input_data], columns=feature_columns)

# Predict button
if st.button("Predict Wine Quality"):
    prediction = predict_wine_quality(input_df)
    st.success(f"Predicted Wine Quality Score: **{prediction[0]}**")




