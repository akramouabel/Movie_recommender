import pandas as pd
import numpy as np
import ast # Used for safely evaluating strings containing Python literal structures (like lists of dicts)
import requests # Essential for making HTTP requests to external APIs (like TMDB)
import pickle # For serializing and deserializing Python objects (saving and loading processed data)
import os # For interacting with the operating system, e.g., creating directories, managing file paths
import time # For pausing execution, crucial for respecting API rate limits
import logging # For structured logging of information, warnings, and errors
import sys
from sklearn.feature_extraction.text import TfidfVectorizer # Converts text data into numerical vectors using TF-IDF
from sklearn.metrics.pairwise import cosine_similarity # Computes cosine similarity between vectors, useful for content-based recommendations

# --- Logging Configuration ---
# Configures the basic logging system. Logs will be printed to the console.
# INFO level logs will show general progress and important steps.
# ERROR/CRITICAL logs will highlight problems.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# --- TMDB API Configuration ---
# Your TMDB API key. This is sensitive information and in a production environment,
# it should ideally be loaded from environment variables (e.g., os.getenv('TMDB_API_KEY'))
# instead of being hardcoded directly in the script for security and flexibility.
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
if not TMDB_API_KEY:
    logging.error("TMDB_API_KEY environment variable not set")
    sys.exit(1)

TMDB_BASE_URL = 'https://api.themoviedb.org/3' # Base URL for TMDB API v3 endpoints

# --- Project Directory Configuration ---
# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Check for Render deployment path
RENDER_DATA_DIR = '/opt/render/project/src/data'
if os.path.exists(RENDER_DATA_DIR):
    PROCESSED_DATA_DIR = RENDER_DATA_DIR
    logging.info(f"Using Render deployment data directory: {PROCESSED_DATA_DIR}")
else:
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    logging.info(f"Using local data directory: {PROCESSED_DATA_DIR}")

# Create data directory if it doesn't exist
try:
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    logging.info(f"Created/verified data directory: {PROCESSED_DATA_DIR}")
except Exception as e:
    logging.error(f"Failed to create data directory: {e}")
    sys.exit(1)

OUTPUT_PKL_FILE = os.path.join(PROCESSED_DATA_DIR, 'movie_data_api.pkl')
logging.info(f"Will save data to: {OUTPUT_PKL_FILE}")

# --- Helper Functions for Data Processing ---

def convert_json_to_list(obj):
    """
    Parses a stringified JSON list of dictionaries (common for 'genres', 'keywords' in raw data)
    and extracts the 'name' value from each dictionary.

    Args:
        obj (str or any): The input string to parse. Expected to be a string
                          representation of a list of dictionaries, e.g., "[{'id': 28, 'name': 'Action'}]".

    Returns:
        list: A list of extracted 'name' strings. Returns an empty list if parsing fails
              or the input is not a valid string.
    """
    names = []
    if isinstance(obj, str):
        try:
            # Safely evaluate the string as a Python literal (list, dict, etc.)
            items = ast.literal_eval(obj)
            if isinstance(items, list): # Ensure the evaluated object is indeed a list
                for item in items:
                    if isinstance(item, dict) and 'name' in item:
                        names.append(item['name'])
        except (ValueError, SyntaxError) as e:
            # Log specific parsing errors but don't stop execution.
            # This allows the script to continue processing other rows.
            logging.debug(f"Could not parse stringified JSON: '{obj}' - Error: {e}")
            pass # Return empty list for malformed strings or non-string inputs
    return names

def get_director_from_crew(crew_list_of_dicts):
    """
    Extracts the director's name from a list of crew dictionaries.
    Assumes 'Director' is a job title within the crew data.

    Args:
        crew_list_of_dicts (list): A list of dictionaries, each representing a crew member.
                                   Expected format: [{'job': 'Director', 'name': 'John Doe'}, ...]

    Returns:
        str or None: The name of the director if found, otherwise None.
    """
    if isinstance(crew_list_of_dicts, list):
        for member in crew_list_of_dicts:
            if isinstance(member, dict) and member.get('job') == 'Director':
                return member.get('name')
    return None # Return None if no director is found or input is invalid

def get_top_n_cast(cast_list_of_dicts, n=3):
    """
    Extracts names of the top N cast members from a list of cast dictionaries.
    The order in the input list determines "top".

    Args:
        cast_list_of_dicts (list): A list of dictionaries, each representing a cast member.
                                   Expected format: [{'name': 'Actor One', 'character': 'Char A'}, ...]
        n (int): The maximum number of cast members to extract.

    Returns:
        list: A list of strings, each being a cast member's name.
    """
    cast_names = []
    if isinstance(cast_list_of_dicts, list):
        for i, member in enumerate(cast_list_of_dicts):
            # Only consider up to 'n' members
            if i < n and isinstance(member, dict) and 'name' in member:
                cast_names.append(member['name'])
            elif i >= n: # Stop once N members are collected
                break
    return cast_names

def get_popular_movies(api_key, page=1):
    """Fetch popular movies from TMDB API."""
    url = f"{TMDB_BASE_URL}/movie/popular"
    params = {
        'api_key': api_key,
        'page': page,
        'language': 'en-US'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()['results']
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching popular movies: {e}")
        return []

def fetch_movie_data_from_api(movie_id, api_key):
    """Fetch detailed movie data from TMDB API."""
    url = f"{TMDB_BASE_URL}/movie/{movie_id}"
    params = {
        'api_key': api_key,
        'append_to_response': 'credits,keywords',
        'language': 'en-US'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        movie_data = response.json()
        
        # Extract required fields
        return {
            'id': movie_data.get('id'),
            'title': movie_data.get('title'),
            'overview': movie_data.get('overview', ''),
            'genres': [genre['name'] for genre in movie_data.get('genres', [])],
            'keywords': [keyword['name'] for keyword in movie_data.get('keywords', {}).get('keywords', [])],
            'cast': [cast['name'] for cast in movie_data.get('credits', {}).get('cast', [])[:5]],
            'crew': [crew['name'] for crew in movie_data.get('credits', {}).get('crew', []) if crew['job'] == 'Director']
        }
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching movie {movie_id}: {e}")
        return None

# --- Main Script Execution ---
# This block ensures that the code inside it only runs when the script is executed directly,
# not when it's imported as a module into another script.
if __name__ == "__main__":
    try:
        logging.info("--- Starting data fetching and processing ---")
        logging.info(f"Using TMDB API key: {TMDB_API_KEY[:4]}...{TMDB_API_KEY[-4:]}")
        logging.info(f"Project root: {PROJECT_ROOT}")
        logging.info(f"Data directory: {PROCESSED_DATA_DIR}")
        logging.info(f"Output file: {OUTPUT_PKL_FILE}")

        # Get popular movies from multiple pages
        all_movie_ids = []
        for page in range(1, 6):  # Get movies from first 5 pages
            logging.info(f"Fetching page {page} of popular movies...")
            popular_movies = get_popular_movies(TMDB_API_KEY, page)
            if not popular_movies:
                logging.error(f"Failed to fetch movies from page {page}")
                continue
            movie_ids = [movie['id'] for movie in popular_movies]
            all_movie_ids.extend(movie_ids)
            logging.info(f"Found {len(movie_ids)} movies on page {page}")
            time.sleep(0.3)  # Respect API rate limits

        if not all_movie_ids:
            logging.error("No movies were fetched. Exiting.")
            sys.exit(1)

        logging.info(f"Found {len(all_movie_ids)} popular movies to process")

        all_fetched_movies_data = []
        total_movies = len(all_movie_ids)

        for i, movie_id in enumerate(all_movie_ids):
            if (i + 1) % 10 == 0 or i == 0:
                logging.info(f"Fetching movie {i + 1}/{total_movies} (ID: {movie_id})...")

            movie_data = fetch_movie_data_from_api(movie_id, TMDB_API_KEY)
            if movie_data:
                all_fetched_movies_data.append(movie_data)

            time.sleep(0.3)

        logging.info(f"Finished fetching data from TMDB API. Fetched data for {len(all_fetched_movies_data)} movies.")

        if not all_fetched_movies_data:
            logging.error("No movies were fetched. Exiting.")
            sys.exit(1)

        # Create DataFrame and process data
        movies_df_processed = pd.DataFrame(all_fetched_movies_data)
        
        logging.info("Processing fetched data for content-based tags and final DataFrame structure...")
        
        for col in ['genres', 'keywords', 'cast', 'crew']:
            movies_df_processed[col] = movies_df_processed[col].apply(
                lambda x: [str(item).replace(" ", "") for item in (x if isinstance(x, list) else []) if item is not None]
            )

        movies_df_processed['overview'] = movies_df_processed['overview'].fillna('')

        movies_df_processed['combined_tags_list'] = \
            movies_df_processed['genres'] * 2 + \
            movies_df_processed['keywords'] + \
            movies_df_processed['cast'] + \
            movies_df_processed['crew'] + \
            movies_df_processed['overview'].apply(lambda x: [str(x)] * 2) + \
            movies_df_processed['title'].apply(lambda x: [str(x).lower().replace(" ", "")])

        movies_df_processed['tags'] = movies_df_processed['combined_tags_list'].apply(
            lambda x: " ".join([str(item) for item in x if item is not None])
        ).apply(lambda x: x.lower().strip())

        # Remove movies with empty tags
        movies_df_processed = movies_df_processed[movies_df_processed['tags'].str.len() > 0]
        
        # Create TF-IDF vectors
        logging.info("Creating TF-IDF vectors...")
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(movies_df_processed['tags'])
        
        # Calculate similarity matrix
        logging.info("Calculating similarity matrix...")
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Save processed data
        logging.info(f"Saving processed data to {OUTPUT_PKL_FILE}...")
        processed_data = {
            'movies_df': movies_df_processed,
            'similarity_matrix': similarity_matrix
        }
        
        try:
            with open(OUTPUT_PKL_FILE, 'wb') as f:
                pickle.dump(processed_data, f)
            logging.info(f"Successfully saved processed data to {OUTPUT_PKL_FILE}")
            
            # Verify the file was created
            if os.path.exists(OUTPUT_PKL_FILE):
                file_size = os.path.getsize(OUTPUT_PKL_FILE)
                logging.info(f"Data file created successfully. Size: {file_size} bytes")
            else:
                logging.error(f"Data file was not created at {OUTPUT_PKL_FILE}")
                sys.exit(1)
        except Exception as e:
            logging.error(f"Error saving data file: {e}")
            sys.exit(1)
        
        logging.info("Data generation completed successfully!")

    except Exception as e:
        logging.error(f"Error during data generation: {e}")
        sys.exit(1)