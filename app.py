# app.py
# Streamlit app for Diabetes Prediction

import pickle
import numpy as np
import streamlit as st

# Page settings
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    with open("diabetes_model.pkl", "rb") as file:
        loaded_model = pickle.load(file)

    # Fix for LogisticRegression multi_class compatibility issue
    if hasattr(loaded_model, "named_steps"):
        if "classifier" in loaded_model.named_steps:
            loaded_model.named_steps["classifier"].multi_class = "ovr"

    return loaded_model


model = load_model()

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient health details below to predict diabetes risk.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)

glucose = st.number_input(
    "Glucose Level",
    min_value=0.0,
    max_value=250.0,
    value=120.0
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0.0,
    max_value=150.0,
    value=70.0
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)

insulin = st.number_input(
    "Insulin",
    min_value=0.0,
    max_value=900.0,
    value=80.0
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

# Prediction
if st.button("Predict"):
    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"Result: Diabetic risk detected. Probability: {probability:.2%}")
    else:
        st.success(f"Result: No diabetic risk detected. Probability: {probability:.2%}")

st.caption("Note: This app is for educational project purposes only, not medical diagnosis.")