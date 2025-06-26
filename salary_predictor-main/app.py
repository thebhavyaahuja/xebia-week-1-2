import streamlit as st
import joblib
import numpy as np
import pandas as pd

model=joblib.load('predict_salary.pkl')
scalar=joblib.load('scaler.pkl')

st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("Salary Predictor")
st.subheader("Predict your salary based on your experience ")
st.write("select the years of experience to predict your salary!")

years= [x for x in range(0, 20)]
years_experience = st.selectbox("Years of Experience", years)

if st.button("Predict Salary"):
    input_data = np.array([[years_experience]])
    input_scaled = scalar.transform(input_data)
    predict_salary = model.predict(input_scaled)
    st.success(f"Predicted Salary: Rs. {predict_salary[0].item():,.2f}")

        