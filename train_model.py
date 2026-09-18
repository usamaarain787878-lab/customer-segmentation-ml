import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle

# 1. Load data
df = pd.read_csv('cleaned_data.csv')

# 2. Industry Standard: Log Transformation for Skewed RFM Data
df['Log_Frequency'] = np.log1p(df['Frequency'])
df['Log_Monetary'] = np.log1p(df['Monetary'])

X = df[['Log_Frequency', 'Log_Monetary']].values

# 3. Train KMeans Model (4 Clusters)
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans.fit(X)

# 4. Save Model
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

print("Enterprise Log-Transformed RFM Model successfully trained and saved!")