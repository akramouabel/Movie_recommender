import os
import sys
import logging
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
import ast # Used for safely evaluating strings containing Python literal structures (like lists of dicts)
import time # For pausing execution, crucial for respecting API rate limits
from sentence_transformers import SentenceTransformer # Added for embedding generation
import torch # Added for model.half()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Get TMDB API key from environment variable
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
if not TMDB_API_KEY:
    logging.error("TMDB_API_KEY not set in environment variables")
    sys.exit(1)

# Base URL for TMDB API
BASE_URL = "https://api.themoviedb.org/3"

# Set up paths
PROJECT_ROOT = Path(__file__).parent.absolute()
RENDER_DATA_DIR = Path("/opt/render/project/src/data")
DATA_DIR = RENDER_DATA_DIR if RENDER_DATA_DIR.exists() else PROJECT_ROOT / "data"
OUTPUT_PKL_FILE = DATA_DIR / "movie_data_api.pkl"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
logging.info(f"Using data directory: {DATA_DIR}")
logging.info(f"Output file will be: {OUTPUT_PKL_FILE}")

def fetch_movies(page=1):
    """Fetch movies from TMDB API with timeout handling"""
    logging.info(f"Attempting to fetch popular movies from TMDB - Page {page}")
    try:
        response = requests.get(
            f"{BASE_URL}/movie/popular",
            params={
                "api_key": TMDB_API_KEY,
                "page": page,
                "language": "en-US"
            },
            timeout=10
        )
        response.raise_for_status()
        time.sleep(0.1) # Small delay to respect API rate limits
        logging.info(f"Successfully fetched page {page}")
        return response.json()["results"]
    except requests.exceptions.Timeout:
        logging.error(f"Timeout while fetching page {page}")
        return []
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching page {page}: {str(e)}")
        return []

def get_movie_details(movie_id):
    """Get detailed movie information with timeout handling"""
    try:
        response = requests.get(
            f"{BASE_URL}/movie/{movie_id}",
            params={
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "append_to_response": "credits,keywords"
            },
            timeout=10
        )
        response.raise_for_status()
        time.sleep(0.1) # Small delay to respect API rate limits
        return response.json()
    except requests.exceptions.Timeout:
        logging.error(f"Timeout while fetching details for movie {movie_id}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching details for movie {movie_id}: {str(e)}")
        return None

def process_movie_data(movie):
    """Process movie data into a standardized format"""
    try:
        return {
            'id': movie.get('id'),
            'title': movie.get('title'),
            'overview': movie.get('overview', ''),
            'genres': [genre['name'] for genre in movie.get('genres', [])],
            'release_date': movie.get('release_date', ''),
            'vote_average': movie.get('vote_average', 0),
            'poster_path': movie.get('poster_path'),
            'cast': [cast['name'] for cast in movie.get('credits', {}).get('cast', [])[:5]],
            'keywords': [keyword['name'] for keyword in movie.get('keywords', {}).get('keywords', [])]
        }
    except Exception as e:
        logging.error(f"Error processing movie {movie.get('id')}: {str(e)}")
        return None

def main():
    """Main function to generate movie data"""
    logging.info("Starting data generation process")
    
    # Fetch movies from first 3 pages only to reduce time
    all_movies = []
    for page in range(1, 241): # Increased to fetch 240 pages for approximately 4800 movies
        logging.info(f"Fetching page {page}")
        movies = fetch_movies(page)
        if not movies:
            logging.warning(f"No movies found on page {page}. Skipping to next page.")
            continue
        all_movies.extend(movies)
        logging.info(f"Found {len(movies)} movies on page {page}. Total movies fetched so far: {len(all_movies)}")
    
    if not all_movies:
        logging.error("No movies fetched from TMDB API. Exiting data generation.")
        sys.exit(1)
    
    logging.info(f"Total raw movies fetched: {len(all_movies)}")
    
    # Process movie data
    processed_movies = []
    for i, movie in enumerate(all_movies):
        if i % 10 == 0: # Log progress every 10 movies
            logging.info(f"Processing movie {i+1}/{len(all_movies)}")
        movie_details = get_movie_details(movie['id'])
        if movie_details:
            processed_movie = process_movie_data(movie_details)
            if processed_movie:
                processed_movies.append(processed_movie)
    
    if not processed_movies:
        logging.error("No movies processed successfully. Exiting data generation.")
        sys.exit(1)
    
    logging.info(f"Total movies successfully processed: {len(processed_movies)}")

    # Create DataFrame
    logging.info("Creating DataFrame from processed movie data.")
    df = pd.DataFrame(processed_movies)
    
    # Create tags for TF-IDF
    logging.info("Generating TF-IDF tags.")
    df['tags'] = df.apply(lambda x: ' '.join([
        str(x['overview']),
        ' '.join(x['genres']),
        ' '.join(x['cast']),
        ' '.join(x['keywords'])
    ]), axis=1)
    
    # Create TF-IDF vectors
    logging.info("Creating TF-IDF vectors.")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['tags'].fillna(''))
    
    # Calculate similarity matrix
    logging.info("Calculating cosine similarity matrix.")
    similarity = cosine_similarity(tfidf_matrix)
    
    # Generate and save movie embeddings
    logging.info("Loading SentenceTransformer model for embedding generation.")
    # Load model and convert to half-precision (FP16) for memory reduction during generation
    embedding_model = SentenceTransformer('msmarco-MiniLM-L6-cos-v5', device='cpu')
    embedding_model.half() # Convert model to FP16

    # Use movie title and overview for embeddings
    texts_for_embeddings = (df['title'] + ' ' + df['overview'].fillna('')).tolist()
    logging.info("Generating movie embeddings.")
    movie_embeddings = embedding_model.encode(texts_for_embeddings, show_progress_bar=True)
    
    embeddings_output_path = DATA_DIR / "movie_embeddings.npy"
    logging.info(f"Attempting to save movie embeddings to {embeddings_output_path}")
    try:
        np.save(embeddings_output_path, movie_embeddings)
        logging.info(f"Movie embeddings saved successfully to {embeddings_output_path}")
    except Exception as e:
        logging.error(f"Error saving movie embeddings: {str(e)}")
        sys.exit(1)

    # Generate and save movie clusters
    logging.info("DEBUG: Starting KMeans clustering process.")
    try:
        from sklearn.cluster import KMeans
        # Use a reasonable number of clusters, e.g., 20, or make it configurable
        kmeans = KMeans(n_clusters=20, random_state=42, n_init=10) # Added n_init to suppress warning
        logging.info("DEBUG: KMeans model initialized. Fitting data...")
        movie_clusters = kmeans.fit_predict(movie_embeddings)
        clusters_output_path = DATA_DIR / "movie_clusters.npy"
        logging.info(f"DEBUG: KMeans clustering complete. Attempting to save clusters to {clusters_output_path}")
        np.save(clusters_output_path, movie_clusters)
        logging.info(f"Movie clusters saved successfully to {clusters_output_path}")
    except Exception as e:
        logging.critical(f"CRITICAL: Error generating or saving movie clusters: {e}", exc_info=True)
        sys.exit(1)

    # Save data
    data = {
        'movies_df': df,
        'similarity_matrix': similarity
    }
    
    logging.info(f"Attempting to save data to {OUTPUT_PKL_FILE}")
    try:
        with open(OUTPUT_PKL_FILE, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Data saved successfully to {OUTPUT_PKL_FILE}")
        
        # Verify file was created
        if OUTPUT_PKL_FILE.exists():
            file_size = OUTPUT_PKL_FILE.stat().st_size
            logging.info(f"File size: {file_size / 1024 / 1024:.2f} MB")
        else:
            logging.error("File was not created successfully after dump.")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"Error saving data: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()