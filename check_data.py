import pickle
import pandas as pd

# Load the data
print("Loading data...")
with open('data/movie_data_api.pkl', 'rb') as f:
    data = pickle.load(f)

# Print the type and keys of the data
print("\nData type:", type(data))
if isinstance(data, dict):
    print("Data keys:", data.keys())
elif isinstance(data, tuple):
    print("Data length:", len(data))
    print("First element type:", type(data[0]))
    if isinstance(data[0], pd.DataFrame):
        movies_df = data[0]
    else:
        print("Unexpected data structure")
        exit(1)
else:
    print("Unexpected data type")
    exit(1)

# Print total number of movies
print(f"\nTotal movies in dataset: {len(movies_df)}")

# Search for Inception
inception_matches = movies_df[movies_df['title'].str.contains('Inception', case=False, na=False)]
print("\nInception matches:")
print(inception_matches[['title', 'release_date']].to_string())

# Print first 10 movies
print("\nFirst 10 movies in dataset:")
print(movies_df[['title', 'release_date']].head(10).to_string()) 