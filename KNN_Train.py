import pandas as pd
import numpy as np

df = pd.read_csv('finaldataset_v0.csv')

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib

features = ['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
       'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo', 'duration_ms', 'time_signature', 'year']

# Extracting features and song IDs
X = df[features]
song_ids = df['id']

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the Nearest Neighbors model
k_neighbors = 1000  # the number of neighbors 
model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
model.fit(X_scaled)

# Save the trained model using joblib
model_filename = 'knn_model.joblib'
joblib.dump(model, model_filename)