import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("heart_model.pkl")

st.title("❤️ Heart Disease Prediction System")

st.write("Enter patient details below:")

# Inputs
age = st.number_input("Age", 1, 100, 50)
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)

# Button
if st.button("Predict"):
    input_data = {
        'age': age,
        'trestbps': trestbps,
        'chol': chol,
        'thalach': thalach,
        'oldpeak': oldpeak,
    }

    input_df = pd.DataFrame([input_data])

    # Match columns
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠️ High chance of Heart Disease")
    else:
        st.success("✅ Low chance of Heart Disease")