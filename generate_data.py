import pandas as pd
import numpy as np
import ast # Used for safely evaluating strings containing Python literal structures (like lists of dicts)
import requests # Essential for making HTTP requests to external APIs (like TMDB)
import pickle # For serializing and deserializing Python objects (saving and loading processed data)
import os # For interacting with the operating system, e.g., creating directories, managing file paths
import time # For pausing execution, crucial for respecting API rate limits
import logging # For structured logging of information, warnings, and errors
from sklearn.feature_extraction.text import TfidfVectorizer # Converts text data into numerical vectors using TF-IDF
from sklearn.metrics.pairwise import cosine_similarity # Computes cosine similarity between vectors, useful for content-based recommendations

# --- Logging Configuration ---
# Configures the basic logging system. Logs will be printed to the console.
# INFO level logs will show general progress and important steps.
# ERROR/CRITICAL logs will highlight problems.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- TMDB API Configuration ---
# Your TMDB API key. This is sensitive information and in a production environment,
# it should ideally be loaded from environment variables (e.g., os.getenv('TMDB_API_KEY'))
# instead of being hardcoded directly in the script for security and flexibility.
TMDB_API_KEY = 'your_tmdb_api_key_here'  # Replace with your actual API key
TMDB_BASE_URL = 'https://api.themoviedb.org/3' # Base URL for TMDB API v3 endpoints

# --- Project Directory Configuration ---
# Defines relative paths for raw input data and processed output data.
# This makes the script more portable as it doesn't rely on absolute paths.
RAW_DATA_DIR = 'raw_data' # Directory where input CSV files are expected (e.g., tmdb_5000_movies.csv)
PROCESSED_DATA_DIR = 'data' # Directory where the final processed .pkl file will be saved

# Constructs full file paths using os.path.join for cross-platform compatibility.
MOVIES_CSV_FILE = os.path.join(RAW_DATA_DIR, 'tmdb_5000_movies.csv')
OUTPUT_PKL_FILE = os.path.join(PROCESSED_DATA_DIR, 'movie_data_api.pkl')

# Ensures that the necessary directories exist. If they don't, they will be created.
# `exist_ok=True` prevents an error if the directory already exists.
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

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

def fetch_movie_data_from_api(movie_id, api_key):
    """
    Fetches comprehensive movie data from the TMDB API for a given movie ID,
    including main details and credits (cast/crew).

    Args:
        movie_id (int): The TMDB ID of the movie.
        api_key (str): Your TMDB API key.

    Returns:
        dict or None: A dictionary containing extracted movie information, or None if
                      the API request fails or response is invalid.
    """
    # Construct the API URL. `append_to_response=credits` fetches cast/crew data in one call.
    url = f'{TMDB_BASE_URL}/movie/{movie_id}?api_key={api_key}&append_to_response=credits'
    try:
        # Make the HTTP GET request with a timeout to prevent indefinite waiting.
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raises an HTTPError for 4xx or 5xx status codes.

        data = response.json() # Parse the JSON response.

        # --- Extract and Process Information from API Response ---
        # Using .get() with a default value prevents KeyError if a key is missing.
        movie_info = {
            'movie_id': data.get('id'),
            'title': data.get('title'),
            'overview': data.get('overview'),
            # TMDB 'genres' are directly a list of dicts. convert_json_to_list expects a string,
            # so we convert it to string for consistent handling if you were using CSV-loaded data.
            # For direct API response, you might process `data.get('genres', [])` directly.
            # However, the current setup is robust for varied inputs.
            'genres': convert_json_to_list(str(data.get('genres', []))),
            # Keywords are nested under 'keywords' -> 'keywords' list of dicts.
            'keywords': convert_json_to_list(str(data.get('keywords', {'keywords': []}).get('keywords', []))),
            'cast': get_top_n_cast(data.get('credits', {}).get('cast', [])),
            # Director is singular, but combined_tags_list expects a list, so wrap it.
            'crew': [get_director_from_crew(data.get('credits', {}).get('crew', []))],
            'poster_path': data.get('poster_path'),
            'release_date': data.get('release_date'),
            # Use 'or 0.0' / 'or 0' to handle None/NaN values by defaulting to 0.
            'vote_average': float(data.get('vote_average') or 0.0),
            'vote_count': int(data.get('vote_count') or 0),
            'popularity': float(data.get('popularity') or 0.0)
        }
        return movie_info
    except requests.exceptions.RequestException as e:
        # Catches all request-related errors (connection, timeout, HTTP errors).
        logging.warning(f"API request failed for movie ID {movie_id}: {e}")
        return None
    except Exception as e:
        # Catches any other unexpected errors during JSON parsing or data extraction.
        logging.error(f"Error processing API response for movie ID {movie_id}: {e}")
        return None

def get_popular_movies(api_key, page=1):
    url = f'{TMDB_BASE_URL}/movie/popular?api_key={api_key}&page={page}'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('results', [])
    except Exception as e:
        logging.error(f"Error fetching popular movies: {e}")
        return []

# --- Main Script Execution ---
# This block ensures that the code inside it only runs when the script is executed directly,
# not when it's imported as a module into another script.
if __name__ == "__main__":
    logging.info("--- Starting data fetching and processing ---")

    # API key validation
    if TMDB_API_KEY == 'your_tmdb_api_key_here':
        logging.critical("ERROR: Please replace 'your_tmdb_api_key_here' with your actual TMDB API key in generate_data.py")
        exit(1)

    # Get popular movies from multiple pages
    all_movie_ids = []
    for page in range(1, 6):  # Get movies from first 5 pages
        popular_movies = get_popular_movies(TMDB_API_KEY, page)
        movie_ids = [movie['id'] for movie in popular_movies]
        all_movie_ids.extend(movie_ids)
        time.sleep(0.3)  # Respect API rate limits

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
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(movies_df_processed['tags'])
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Save processed data
    processed_data = {
        'movies_df': movies_df_processed,
        'similarity_matrix': similarity_matrix
    }
    
    with open(OUTPUT_PKL_FILE, 'wb') as f:
        pickle.dump(processed_data, f)
    
    logging.info(f"Successfully saved processed data to {OUTPUT_PKL_FILE}")

    logging.info("Data generation completed successfully!")