import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model('diabetes_model.h5')
scaler = joblib.load('Scaler.pkl')

#app page
st.set_page_config(page_title="Diabetes Prediction App", page_icon=":hospital:", layout="centered")
st.title("Diabetes Prediction App ")
st.markdown("Enter the following details to predict diabetes:")
# Input fields for user data
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, value=0)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, value=0)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, value=0)
insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, value=0)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, value=0.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.0)
age = st.number_input("Age", min_value=0, max_value=120, value=0)

# Button to trigger prediction
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    prediction = (prediction > 0.5).astype(int)
    prediction_text = "Diabetic" if prediction[0][0] == 1 else "Not Diabetic"
    st.success(f"The model predicts that the patient is: {prediction_text}")
