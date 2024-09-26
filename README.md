
# Song Recommendation System

This project implements a song recommendation system using K-Nearest Neighbors (KNN) based on various audio features extracted from a dataset. The recommendation model predicts similar songs based on a given song ID.

## Requirements

Make sure you have the following Python packages installed:

- pandas
- numpy
- scikit-learn
- joblib

You can install these packages using pip:

```bash
pip install pandas numpy scikit-learn joblib
```



# Dataset

The project requires two CSV files:

1. **finaldataset_v0.csv** : This file should contain song features including the following columns:

	- explicit
	- danceability
	- energy
	- key
	- loudness
	- mode
	- speechiness
	- acousticness
	- instrumentalness
	- liveness
	- valence
	- tempo
	- duration_ms
	- time_signature
	- year
	- id (unique identifier for each song)

2. **1mDataset.csv** : This file contains additional song details, which should include at least the name and id columns.

## Example

To get recommendations for a specific song, replace `input_song_id` with the ID of the song you want recommendations for and run the script.

`input_song_id = 'YOUR_SONG_ID_HERE'`

## Conclusion

This project demonstrates how to build a simple song recommendation system using KNN. You can enhance the model by exploring different algorithms or adding more features to improve recommendation quality.
