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

# --- Main Script Execution ---
# This block ensures that the code inside it only runs when the script is executed directly,
# not when it's imported as a module into another script.
if __name__ == "__main__":
    logging.info("--- Starting data fetching and processing ---")

    # Load movie IDs from the local CSV file. This CSV primarily provides a list of movie IDs
    # and titles to iterate through, as the comprehensive data is fetched from TMDB API.
    try:
        # Only load 'id' and 'title' columns to save memory if CSV is large.
        movies_ids_df_raw = pd.read_csv(MOVIES_CSV_FILE, usecols=['id', 'title'])
        # Get unique movie IDs to avoid redundant API calls if CSV has duplicates.
        movie_ids_to_fetch = movies_ids_df_raw['id'].unique().tolist()
        logging.info(f"Loaded {len(movie_ids_to_fetch)} unique movie IDs from {MOVIES_CSV_FILE}.")
    except FileNotFoundError:
        logging.critical(f"ERROR: Raw movies CSV not found at {MOVIES_CSV_FILE}. Please check your '{RAW_DATA_DIR}' folder.")
        exit(1) # Exit with an error code
    except Exception as e:
        logging.critical(f"ERROR loading raw movies CSV: {e}")
        exit(1) # Exit with an error code

    all_fetched_movies_data = []
    total_movies = len(movie_ids_to_fetch)
    logging.info(f"Attempting to fetch comprehensive data for {total_movies} movies from TMDB API.")

    # API key validation before starting expensive API calls.
    if TMDB_API_KEY == 'your_tmdb_api_key_here':
        logging.critical("ERROR: Please replace 'your_tmdb_api_key_here' with your actual TMDB API key in generate_data.py")
        exit(1)

    # Iterate through each movie ID and fetch data from TMDB API.
    for i, movie_id in enumerate(movie_ids_to_fetch):
        # Log progress periodically to give feedback on long-running operations.
        if (i + 1) % 100 == 0 or i == 0:
            logging.info(f"Fetching movie {i + 1}/{total_movies} (ID: {movie_id})...")

        movie_data = fetch_movie_data_from_api(movie_id, TMDB_API_KEY)
        if movie_data:
            all_fetched_movies_data.append(movie_data)

        # Introduce a small delay to respect TMDB API rate limits.
        # TMDB's public API usually allows ~40 requests per 10 seconds.
        # A 0.3 second delay ensures you stay well within limits (1/0.3 = ~3.3 requests per second).
        time.sleep(0.3)

    logging.info(f"Finished fetching data from TMDB API. Fetched data for {len(all_fetched_movies_data)} movies.")

    # Create a Pandas DataFrame from the list of fetched movie dictionaries.
    movies_df_processed = pd.DataFrame(all_fetched_movies_data)
    
    # --- Post-fetching processing for content-based 'tags' ---
    logging.info("Processing fetched data for content-based tags and final DataFrame structure...")
    
    # Ensure list-like columns are properly handled (NaN values filled with empty lists)
    # and that individual elements within these lists are strings with spaces removed.
    # Removing spaces from multi-word tags (e.g., "Science Fiction" -> "ScienceFiction")
    # helps TF-IDF treat them as single tokens, which is often desired for genre/keyword matching.
    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies_df_processed[col] = movies_df_processed[col].apply(
            lambda x: [str(item).replace(" ", "") for item in (x if isinstance(x, list) else []) if item is not None]
        )

    # Fill any missing overviews with an empty string. This prevents errors when concatenating text.
    movies_df_processed['overview'] = movies_df_processed['overview'].fillna('')

    # Combine all relevant features into a single list of strings per movie.
    # Features are duplicated (`* 2`) to give them more weight during TF-IDF vectorization.
    # The title is also added (lowercase, no spaces) to ensure direct title matches contribute.
    movies_df_processed['combined_tags_list'] = \
        movies_df_processed['genres'] * 2 + \
        movies_df_processed['keywords'] + \
        movies_df_processed['cast'] + \
        movies_df_processed['crew'] + \
        movies_df_processed['overview'].apply(lambda x: [str(x)] * 2) + \
        movies_df_processed['title'].apply(lambda x: [str(x).lower().replace(" ", "")])

    # Convert the list of combined tags into a single space-separated string.
    # All tags are converted to lowercase and leading/trailing spaces are stripped.
    movies_df_processed['tags'] = movies_df_processed['combined_tags_list'].apply(
        lambda x: " ".join([str(item) for item in x if item is not None])
    ).apply(lambda x: x.lower().strip())

    # --- Data Cleaning: Remove movies with empty tags ---
    # Movies without meaningful 'tags' (e.g., no overview, genres, cast, etc.) cannot be
    # accurately used for similarity calculation. They are removed here.
    initial_count = len(movies_df_processed)
    movies_df_processed = movies_df_processed[movies_df_processed['tags'].str.strip() != ''].copy()
    if len(movies_df_processed) < initial_count:
        logging.warning(f"Removed {initial_count - len(movies_df_processed)} movies due to empty or missing 'tags' data.")
    
    # --- TF-IDF Vectorization and Cosine Similarity Calculation ---
    logging.info("Generating TF-IDF matrix...")
    # TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numerical vectors.
    # `stop_words='english'` removes common English words (like 'the', 'a', 'is') that don't
    # contribute much to meaning.
    # `min_df=5` ignores terms that appear in fewer than 5 documents. This helps remove rare
    # or misspelled words that might be noise, focusing on more significant features.
    tfidf = TfidfVectorizer(stop_words='english', min_df=5)
    tfidf_matrix = tfidf.fit_transform(movies_df_processed['tags']) # Fit and transform the 'tags' column

    logging.info("Calculating cosine similarity matrix...")
    # Cosine similarity measures the cosine of the angle between two non-zero vectors.
    # It determines how similar two documents (movies, in this case) are based on their TF-IDF vectors.
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # --- Save Processed Data ---
    logging.info(f"Saving processed data (DataFrame and similarity matrix) to {OUTPUT_PKL_FILE}...")
    try:
        # Use pickle to serialize and save the DataFrame and similarity matrix as a tuple.
        # This allows the backend to load them quickly without reprocessing.
        with open(OUTPUT_PKL_FILE, 'wb') as file:
            pickle.dump((movies_df_processed, cosine_sim), file)
        logging.info("--- Data fetching, processing, and saving complete! ---")
        logging.info(f"Successfully processed and saved data for {len(movies_df_processed)} movies to {OUTPUT_PKL_FILE}.")
        logging.info("First 5 rows of the processed DataFrame (including new 'tags' field):")
        logging.info(movies_df_processed.head(5).T) # Transpose for better readability in logs
    except Exception as e:
        logging.critical(f"ERROR: Could not save PKL file to {OUTPUT_PKL_FILE}: {e}")
        exit(1) # Exit with an error code if saving fails

    logging.info("Data generation completed successfully!")