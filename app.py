from flask import Flask, request, jsonify
import os
import logging
import pandas as pd
import json
from backend import recommender
from flask_cors import CORS
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Flask Application Initialization ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Global variables for data
movie_data = None
similarity_matrix = None

def load_data():
    """Load data with memory optimization"""
    global movie_data, similarity_matrix
    
    try:
        data_path = 'data/movie_data_api.pkl'  # Hardcoded path
        logger.info(f"Attempting to load recommendation data from: {data_path}")
        
        # Load data in chunks if it's a large file
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        # Extract only necessary columns to reduce memory usage
        movie_data = pd.DataFrame({
            'id': data['id'],
            'title': data['title'],
            'overview': data['overview'],
            'poster_path': data['poster_path'],
            'release_date': data['release_date'],
            'vote_average': data['vote_average'],
            'genres': data['genres']
        })
        
        # Convert similarity matrix to sparse format if it exists
        if 'similarity_matrix' in data:
            similarity_matrix = data['similarity_matrix']
            if isinstance(similarity_matrix, np.ndarray):
                from scipy import sparse
                similarity_matrix = sparse.csr_matrix(similarity_matrix)
        
        # Clear memory
        del data
        gc.collect()
        
        logger.info("Data loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return False

# Load data when the application starts
load_data()

# --- API Routes ---

@app.route('/')
def home():
    """
    Root endpoint to confirm the backend is running.
    Returns a simple status message.
    """
    return jsonify({"message": "Movie Recommendation API is running!"})

@app.route('/recommend', methods=['GET'])
def recommend_movie():
    """
    Handles movie recommendation requests from the frontend.
    Accepts query parameters for title, number of recommendations, year range, and genres.
    Returns a JSON response with recommendations or an error message.
    """
    movie_title = request.args.get('title')
    top_n = request.args.get('top_n', 10, type=int)
    min_year = request.args.get('min_year', type=int)
    max_year = request.args.get('max_year', type=int)
    selected_genres_raw_str = request.args.get('genres') 

    selected_genres_list = []
    if selected_genres_raw_str:
        try:
            temp_genres = json.loads(selected_genres_raw_str)
            if isinstance(temp_genres, list):
                selected_genres_list = [g.strip().replace(" ", "") for g in temp_genres if isinstance(g, str) and g.strip()]
            else:
                logger.warning(f"Expected list for genres after JSON decode but got {type(temp_genres)}: {temp_genres}")
                selected_genres_list = [g.strip().replace(" ", "") for g in selected_genres_raw_str.split(',') if g.strip()]
        except json.JSONDecodeError:
            logger.error(f"Failed to decode genres JSON: {selected_genres_raw_str}. Treating as no genres selected.")
            selected_genres_list = []
        except Exception as e:
            logger.error(f"Unexpected error processing genres: {e}")
            selected_genres_list = []
    logger.info(f"Recommendation request for title: '{movie_title}', min_year: {min_year}, max_year: {max_year}, genres: {selected_genres_list}")

    if not movie_title:
        return jsonify({"error": "Movie title parameter 'title' is required."}), 400

    recommendations, matched_title = recommender.get_recommendations(
        movie_title=movie_title,
        num_recommendations=top_n,
        min_year=min_year,
        max_year=max_year,
        selected_genres=selected_genres_list
    )

    if recommendations is None:
        logger.error(f"Recommender returned None for '{movie_title}': {matched_title}")
        return jsonify({"error": matched_title}), 500
    elif not recommendations:
        logger.info(f"No recommendations found for '{movie_title}' with applied filters. Message: {matched_title}")
        return jsonify({"recommendations": [], "message": matched_title}), 200
    else:
        logger.info(f"Successfully retrieved {len(recommendations)} recommendations for '{movie_title}'.")
        return jsonify({"recommendations": recommendations, "matched_title": matched_title}), 200

@app.route('/suggest', methods=['GET'])
def suggest_movie():
    """
    Provides movie title suggestions based on a partial query string.
    Returns a list of close matches for autocomplete or user assistance.
    """
    query = request.args.get('query')
    if not query:
        return jsonify({"suggestions": []}), 200
    suggestions = recommender.get_title_suggestions(query)
    logger.info(f"Suggestions for '{query}': {suggestions}")
    return jsonify({"suggestions": suggestions}), 200

@app.route('/genres', methods=['GET'])
def genres():
    """
    Returns a list of all unique movie genres available in the dataset.
    Used to populate filter options in the frontend.
    """
    if recommender.movies_df is None:
        logger.error("movies_df not loaded in recommender module when /genres was requested. Data pre-loading failed.")
        return jsonify({"error": "Movie data not loaded on backend. Please ensure data generation script was run."}), 500
    all_genres = []
    for movie_genres in recommender.movies_df['genres'].dropna():
        if isinstance(movie_genres, list):
            all_genres.extend(movie_genres)
    unique_genres = sorted(list(set(all_genres)))
    logger.info(f"Fetched {len(unique_genres)} unique genres.")
    return jsonify({"genres": unique_genres})

@app.route('/years', methods=['GET'])
def years():
    """
    Returns a list of all unique movie release years available in the dataset.
    Used to populate filter options in the frontend.
    """
    if recommender.movies_df is None:
        logger.error("movies_df not loaded in recommender module when /years was requested. Data pre-loading failed.")
        return jsonify({"error": "Movie data not loaded on backend. Please ensure data generation script was run."}), 500
    years_list = []
    for date_str in recommender.movies_df['release_date'].dropna():
        converted_date = pd.to_datetime(date_str, errors='coerce')
        if pd.notna(converted_date): 
            years_list.append(converted_date.year)
    unique_years = sorted(list(set(years_list)))
    logger.info(f"Fetched {len(unique_years)} unique years.")
    return jsonify({"years": unique_years})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001))) 