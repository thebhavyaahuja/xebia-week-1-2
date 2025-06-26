import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
import streamlit as st

# load the saved model
kmeans= joblib.load('kmeans_model.pkl')
df=pd.read_csv('Mall_Customers.csv')
df.drop(['CustomerID','Age','Gender'], axis=1, inplace=True)
X_array = df.values

# Streamlit app
st.set_page_config(page_title="Customer Cluster Prediction", layout="centered")
st.title("Customer Cluster Prediction")
st.write("Enter the annual income and spending score to predict the customer cluster.")

# inputs
annual_income = st.number_input("Annual Income (k$)", min_value=0, max_value=400, value=100)
spending_score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=20)

# predict cluster
if st.button("Predict Cluster"):
    input_data = np.array([[annual_income, spending_score]])
    cluster = kmeans.predict(input_data)[0]
    st.success(f"The predicted cluster for the customer is: Cluster {cluster}")

