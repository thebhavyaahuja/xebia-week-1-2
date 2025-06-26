import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv('Mall_Customers.csv')
df.head()
df.drop(['CustomerID','Gender','Age'], axis=1, inplace=True)

wcss_list=[]
from sklearn.cluster import KMeans
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=1)
    kmeans.fit(df)
    wcss_list.append(kmeans.inertia_)

# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 11), wcss_list, marker='o')
# plt.title('Elbow Method for Optimal k')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('WCSS')
# plt.grid()
# plt.show()

#print max value of income in dataset
# print("Maximum annual income in dataset:", df['Annual Income (k$)'].max())

model= KMeans(n_clusters=6, init='k-means++', random_state=1)
y_pred = model.fit_predict(df)

X_array = df.values

# plotting graph of clusters
plt.scatter(X_array[y_pred == 0, 0], X_array[y_pred == 0, 1], s=100, c='Green', label='Cluster 1')
plt.scatter(X_array[y_pred == 1, 0], X_array[y_pred == 1, 1], s=100, c='Red', label='Cluster 2')
plt.scatter(X_array[y_pred == 2, 0], X_array[y_pred == 2, 1], s=100, c='Yellow', label='Cluster 3')
plt.scatter(X_array[y_pred == 3, 0], X_array[y_pred == 3, 1], s=100, c='Blue', label='Cluster 4')
plt.scatter(X_array[y_pred == 4, 0], X_array[y_pred == 4, 1], s=100, c='Orange', label='Cluster 5')
plt.scatter(X_array[y_pred == 5, 0], X_array[y_pred == 5, 1], s=100, c='Pink', label='Cluster 6')
plt.title('Customer Segmentation Graph')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

joblib.dump(model, 'kmeans_model.pkl')
print("Model saved as kmeans_model.pkl")