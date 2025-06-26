#a simple app to predict the salary based on the number of years of experiences
#import libraries
#joblib is used to save the created model model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

#loding the data set into the dataframe
df = pd.read_csv("salary_data.csv")
#print(df.info())
#split the data into target variable and Independent variable
X=df[["YearsExperience"]]
y=df[["Salary"]]

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
#Scale down the data
#creating object of StandardScaler module to scale down the data
scaler= StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.fit_transform(X_test)

#Train the model
model=LinearRegression()
model.fit(X_train_scaled, y_train)
joblib.dump(model, 'predict_salary.pkl')
joblib.dump(scaler, 'scaler.pkl')
print ("Model and scaler are saved.")