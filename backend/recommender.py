# backend/recommender.py
# Core Recommendation Engine for Movie Recommender System
# ------------------------------------------------------
# This module loads movie data, computes similarities, and generates recommendations.
# It supports multi-feature similarity (text, genre, year), embeddings, clustering, and filtering.

import pandas as pd
import numpy as np
import ast # For safely evaluating string representations of Python literal structures (e.g., lists from CSV)
import requests # Not directly used here, but may be useful for future extensions
import pickle # Crucial for deserializing (loading) the pre-processed data
import os # For interacting with the operating system, particularly for path manipulation
import time # Not directly used here, but may be useful for profiling
import logging # For structured logging
from difflib import get_close_matches # For fuzzy string matching
import sys # Added for sys.exit()
# from sentence_transformers import SentenceTransformer # Removed: only needed for data generation
# import torch # Removed: only needed for data generation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration for PKL file path and data components ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
PKL_FILE_PATH = os.path.join(project_root_dir, 'data', 'movie_data_api.pkl')
EMBEDDINGS_FILE_PATH = os.path.join(project_root_dir, 'data', 'movie_embeddings.npy')
CLUSTERS_FILE_PATH = os.path.join(project_root_dir, 'data', 'movie_clusters.npy')

# --- Global Variables for Loaded Data ---
movies_df = None # Pandas DataFrame containing all movie metadata
cosine_sim = None # Cosine similarity matrix (legacy, not used with embeddings)
all_titles_for_suggestions = [] # List of all movie titles for suggestions
movie_embeddings = None # Numpy array of movie embeddings
movie_clusters = None # Numpy array of cluster assignments

# --- Helper Functions for Multi-Feature Similarity ---
def genre_similarity(genres1, genres2):
    """
    Computes Jaccard similarity between two genre lists.
    Returns a float between 0 and 1.
    """
    set1, set2 = set(genres1), set(genres2)
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def year_similarity(year1, year2):
    """
    Computes similarity between two years (1 if same, 0 if 10+ years apart).
    Returns a float between 0 and 1.
    """
    try:
        y1, y2 = int(year1), int(year2)
        return 1.0 - min(abs(y1 - y2), 10) / 10  # 1 if same year, 0 if 10+ years apart
    except:
        return 0.0

def combined_similarity(idx1, idx2, embeddings, movies_df, w_text=0.6, w_genre=0.3, w_year=0.1):
    """
    Computes a weighted similarity score between two movies using text, genre, and year.
    """
    text_sim = cosine_similarity([embeddings[idx1]], [embeddings[idx2]])[0][0]
    genre_sim = genre_similarity(movies_df.iloc[idx1]['genres'], movies_df.iloc[idx2]['genres'])
    year1 = str(movies_df.iloc[idx1]['release_date']).split('-')[0]
    year2 = str(movies_df.iloc[idx2]['release_date']).split('-')[0]
    year_sim = year_similarity(year1, year2)
    return w_text * text_sim + w_genre * genre_sim + w_year * year_sim

# --- Data Component Loading Utilities ---
def load_embeddings():
    """
    Loads movie embeddings from disk.
    Returns a numpy array of embeddings.
    """
    global movie_embeddings
    if movie_embeddings is not None:
        logging.info("Movie embeddings already loaded.")
        return movie_embeddings

    if os.path.exists(EMBEDDINGS_FILE_PATH):
        try:
            movie_embeddings = np.load(EMBEDDINGS_FILE_PATH)
            logging.info(f"Successfully loaded movie embeddings from {EMBEDDINGS_FILE_PATH}")
            return movie_embeddings
        except Exception as e:
            logging.critical(f"CRITICAL: Error loading embeddings file {EMBEDDINGS_FILE_PATH}: {e}")
            sys.exit(1) # Exit if embeddings cannot be loaded
    else:
        logging.critical(f"CRITICAL: Embeddings file NOT FOUND at {EMBEDDINGS_FILE_PATH}. "
                         "Ensure 'generate_data.py' was run successfully during the build phase.")
        sys.exit(1) # Exit if embeddings file is missing

def load_clusters(n_clusters=20):
    """
    Loads cluster assignments from disk.
    Returns a numpy array of cluster labels.
    """
    global movie_clusters
    if movie_clusters is not None:
        return movie_clusters
    if os.path.exists(CLUSTERS_FILE_PATH):
        try:
            movie_clusters = np.load(CLUSTERS_FILE_PATH)
            logging.info(f"Successfully loaded movie clusters from {CLUSTERS_FILE_PATH}")
            return movie_clusters
        except Exception as e:
            logging.critical(f"CRITICAL: Error loading clusters file {CLUSTERS_FILE_PATH}: {e}")
            sys.exit(1) # Exit if clusters cannot be loaded
    else:
        logging.critical(f"CRITICAL: Clusters file NOT FOUND at {CLUSTERS_FILE_PATH}. "
                         "Ensure 'generate_data.py' was run successfully during the build phase.")
        sys.exit(1) # Exit if clusters file is missing

# --- Data Loading ---
def load_recommendation_data():
    """
    Loads the pre-processed movie DataFrame, embeddings, clusters, and similarity matrix from PKL and NPY files.
    Ensures the 'genres' column is properly parsed as lists.
    Returns True if successful, False otherwise.
    """
    global movies_df, cosine_sim, all_titles_for_suggestions
    if movies_df is not None and cosine_sim is not None:
        logging.info("Recommendation data already loaded in recommender module (skipped reload).")
        return True
    logging.info(f"Attempting to load recommendation data from: {PKL_FILE_PATH}")
    try:
        if not os.path.exists(PKL_FILE_PATH):
            logging.error(f"Error: PKL file NOT FOUND at {PKL_FILE_PATH}. Please ensure 'generate_data.py' was run successfully.")
            return False
        with open(PKL_FILE_PATH, 'rb') as file:
            data = pickle.load(file)
            # Handle both dict and tuple formats
            if isinstance(data, dict):
                movies_df = data['movies_df']
                cosine_sim = data['similarity_matrix']
            elif isinstance(data, tuple):
                movies_df = data[0]
                cosine_sim = data[1] if len(data) > 1 else None
            else:
                logging.error(f"Unrecognized data format in PKL file: {type(data)}")
                return False
        # Ensure 'genres' column contains actual lists (fix for data parsing inconsistencies)
        if 'genres' in movies_df.columns:
            # Apply ast.literal_eval only to those that are strings
            movies_df['genres'] = movies_df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            # Ensure all entries are lists, default to empty list if None/NaN
            movies_df['genres'] = movies_df['genres'].apply(lambda x: [item.strip() for item in x] if isinstance(x, list) else [])
        all_titles_for_suggestions = movies_df['title'].tolist()
        logging.info(f"First 5 movie titles loaded: {all_titles_for_suggestions[:5]}")
        logging.info(f"Successfully loaded {len(movies_df)} movies and cosine similarity matrix.")
        if movies_df is None or movies_df.empty or cosine_sim is None or (hasattr(cosine_sim, 'size') and cosine_sim.size == 0):
            logging.error("Loaded data is empty or malformed after pickle.load. Clearing global variables.")
            movies_df = None
            cosine_sim = None
            all_titles_for_suggestions = []
            return False
        # --- Load pre-computed data components ---
        load_embeddings()
        load_clusters()
        logging.info("Recommendation data loading process completed successfully.")
        return True
    except Exception as e:
        logging.error(f"EXCEPTION during recommendation data loading: {e}", exc_info=True)
        movies_df = None
        cosine_sim = None
        all_titles_for_suggestions = []
        return False

# Call to load data when the module is imported
load_recommendation_data()

def get_title_suggestions(title_query, n=5, cutoff=0.3):
    """
    Returns a list of movie titles that are "close" to the provided query string using fuzzy matching.
    Used for autocomplete and user assistance in the frontend.
    """
    if not all_titles_for_suggestions:
        logging.warning("Cannot provide suggestions, movie titles list is empty. Data might not be loaded.")
        return []
    logging.info(f"DEBUG: get_title_suggestions received query: '{title_query}' (cutoff={cutoff})")
    suggestions = get_close_matches(title_query, all_titles_for_suggestions, n=n, cutoff=cutoff)
    logging.info(f"DEBUG: get_close_matches returned suggestions for '{title_query}': {suggestions}")
    return suggestions

def get_recommendations(movie_title, num_recommendations=10, min_year=None, max_year=None, selected_genres=None):
    """
    Generates a list of movie recommendations based on a given movie title and optional filters.
    Uses multi-feature similarity (text, genre, year) and cluster-based diversity.
    Returns a list of formatted movie dicts and the matched title.
    """
    if not load_recommendation_data():
        return None, "Recommendation system data not loaded or failed to load on backend."
    global movies_df, movie_embeddings, movie_clusters
    all_titles = movies_df['title'].tolist()
    if not movie_title:
        return [], "Please enter a movie title to get recommendations."
    closest_matches = get_close_matches(movie_title, all_titles, n=1, cutoff=0.6)
    if not closest_matches:
        logging.info(f"No close match found in the dataset for input: '{movie_title}'.")
        return [], f"Sorry, '{movie_title}' was not found in our database. Please try another title."
    exact_matched_title = closest_matches[0]
    logging.info(f"Matched input '{movie_title}' to '{exact_matched_title}'.")
    try:
        movie_idx = movies_df[movies_df['title'] == exact_matched_title].index[0]
    except IndexError:
        logging.error(f"Error: Matched title '{exact_matched_title}' not found in DataFrame index after fuzzy match.")
        return [], "An internal error occurred while finding the movie for recommendations. Please try again."

    # Define desired columns for the recommendation response
    desired_columns = ['id', 'title', 'release_date', 'genres', 'overview', 'poster_path'] # Assuming these columns exist in movies_df

    # --- Multi-feature similarity with diversity ---
    similarities = []
    for idx in range(len(movies_df)):
        if idx == movie_idx:
            continue
        # --- Year Filter Logic ---
        rec_movie = movies_df.iloc[idx]
        release_year = None
        if pd.notna(rec_movie.get('release_date')) and isinstance(rec_movie['release_date'], str):
            try:
                release_year = int(str(rec_movie['release_date']).split('-')[0])
            except ValueError:
                pass
        if min_year is not None and (release_year is None or release_year < min_year):
            continue
        if max_year is not None and (release_year is None or release_year > max_year):
            continue
        # --- Genre Filter Logic ---
        parsed_selected_genres = [g.strip() for g in selected_genres if isinstance(g, str)] if selected_genres else []
        if parsed_selected_genres:
            movie_genres_from_df = rec_movie.get('genres', [])
            if not isinstance(movie_genres_from_df, list):
                continue
            movie_genres_set = {g.strip() for g in movie_genres_from_df if isinstance(g, str)}
            if not any(s_genre in movie_genres_set for s_genre in parsed_selected_genres):
                continue
        sim = combined_similarity(movie_idx, idx, movie_embeddings, movies_df)
        similarities.append((idx, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)

    # --- Cluster-based diversity and detailed recommendation formatting ---
    seen_clusters = set()
    filtered_recommendations = []
    recommended_movie_ids = set()

    for idx, sim in similarities:
        cluster = movie_clusters[idx]
        if cluster not in seen_clusters:
            rec_movie_series = movies_df.iloc[idx] # This is a pandas Series representing the movie row
            if rec_movie_series['id'] not in recommended_movie_ids:
                # Construct a dictionary with specific details for the frontend
                movie_details = {}
                for col in desired_columns:
                    movie_details[col] = rec_movie_series.get(col)

                # Special handling for 'release_date' to ensure it's just the year
                raw_release_date = movie_details.get('release_date')
                if pd.notna(raw_release_date) and isinstance(raw_release_date, str):
                    try:
                        movie_details['release_date'] = str(pd.to_datetime(raw_release_date).year)
                    except (ValueError, TypeError):
                        movie_details['release_date'] = None # Or keep original string if it's already just the year or invalid
                else:
                    movie_details['release_date'] = None

                # Ensure 'genres' is always a list, even if empty or initially not a list
                if not isinstance(movie_details.get('genres'), list):
                    movie_details['genres'] = []
                else:
                    # Clean up genres (strip whitespace, ensure all are strings)
                    movie_details['genres'] = [g.strip() for g in movie_details['genres'] if isinstance(g, str)]

                filtered_recommendations.append(movie_details)
                recommended_movie_ids.add(rec_movie_series['id'])
                seen_clusters.add(cluster) # Add to seen clusters here after processing movie

                if len(filtered_recommendations) >= num_recommendations:
                    break # Stop if we have enough recommendations

    return filtered_recommendations, exact_matched_title