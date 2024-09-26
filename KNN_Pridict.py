import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('finaldataset_v0.csv')
allFeaturedDataset = pd.read_csv('1mDataset.csv')

# Define the features used for the model
features = ['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness', 'liveness',
            'valence', 'tempo', 'duration_ms', 'time_signature', 'year']

# Extract the features and song IDs
X = df[features]
song_ids = df['id']

# Scale the features (use the same scaler that was used during model training)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load the trained NearestNeighbors model from the .joblib file
model = joblib.load('knn_model.joblib')

# Model Predictions Function
def recommendSong(songId):
    # Find the index of the song in the dataset
    song_index = df[df['id'] == songId].index[0] 
    query_song_features = X_scaled[song_index].reshape(1, -1)
    
    # Use the model to find the nearest neighbors
    distances, indices = model.kneighbors(query_song_features)
    
    # Get the recommended song IDs and their details
    recommended_song_ids = song_ids.iloc[indices[0]].values.tolist()
    recommendations = allFeaturedDataset[allFeaturedDataset['id'].isin(recommended_song_ids)][['name', 'id']]
    
    # Add the distances to the recommendations and sort by distance
    recommendations['distance'] = distances[0]
    recommendations = recommendations.sort_values(by='distance', ascending=True)
    
    return recommendations

# Input song ID for predictions
input_song_id = '1wsRitfRRtWyEapl0q22o8'
recommended_songs = recommendSong(input_song_id)

print("Recommended Songs:")
print(recommended_songs)













