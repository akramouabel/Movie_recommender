# backend/main.py
# Flask Backend for Movie Recommendation System
# ---------------------------------------------
# This file defines the main API endpoints for the movie recommender backend.
# It handles requests from the frontend, applies filters, and returns recommendations or metadata.

from flask import Flask, request, jsonify
import os
import logging
import pandas as pd
import json
from . import recommender
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
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        # Handle both dict and tuple formats
        if isinstance(data, dict):
            movie_data = pd.DataFrame({
                'id': data['id'],
                'title': data['title'],
                'overview': data['overview'],
                'poster_path': data['poster_path'],
                'release_date': data['release_date'],
                'vote_average': data['vote_average'],
                'genres': data['genres']
            })
            if 'similarity_matrix' in data:
                similarity_matrix = data['similarity_matrix']
                if isinstance(similarity_matrix, np.ndarray):
                    from scipy import sparse
                    similarity_matrix = sparse.csr_matrix(similarity_matrix)
        elif isinstance(data, tuple):
            movie_data = data[0]
            similarity_matrix = data[1] if len(data) > 1 else None
        else:
            logger.error(f"Unrecognized data format in PKL file: {type(data)}")
            return False
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

@app.route('/api/movies', methods=['GET'])
def get_movies():
    try:
        if movie_data is None:
            return jsonify({"error": "Movie data not loaded"}), 500
            
        # Get pagination parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        # Calculate start and end indices
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        # Get paginated data
        paginated_data = movie_data.iloc[start_idx:end_idx]
        
        # Convert to list of dictionaries
        movies = paginated_data.to_dict('records')
        
        # Add total count for pagination
        total_movies = len(movie_data)
        
        return jsonify({
            "movies": movies,
            "total": total_movies,
            "page": page,
            "per_page": per_page,
            "total_pages": (total_movies + per_page - 1) // per_page
        })
    except Exception as e:
        logger.error(f"Error in get_movies: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/movies/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    try:
        if movie_data is None:
            return jsonify({"error": "Movie data not loaded"}), 500
            
        movie = movie_data[movie_data['id'] == movie_id]
        if movie.empty:
            return jsonify({"error": "Movie not found"}), 404
            
        return jsonify(movie.iloc[0].to_dict())
    except Exception as e:
        logger.error(f"Error in get_movie: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommendations/<int:movie_id>', methods=['GET'])
def get_recommendations(movie_id):
    try:
        if movie_data is None or similarity_matrix is None:
            return jsonify({"error": "Data not loaded"}), 500
            
        # Get the index of the movie
        movie_idx = movie_data[movie_data['id'] == movie_id].index
        if len(movie_idx) == 0:
            return jsonify({"error": "Movie not found"}), 404
            
        # Get similarity scores for the movie
        movie_similarities = similarity_matrix[movie_idx[0]].toarray()[0]
        
        # Get indices of top similar movies
        similar_indices = np.argsort(movie_similarities)[::-1][1:11]  # Top 10 similar movies
        
        # Get the similar movies data
        similar_movies = movie_data.iloc[similar_indices].to_dict('records')
        
        return jsonify({"recommendations": similar_movies})
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search_movies():
    try:
        if movie_data is None:
            return jsonify({"error": "Movie data not loaded"}), 500
            
        query = request.args.get('query', '').lower()
        if not query:
            return jsonify({"movies": []}), 200
            
        # Search in titles
        results = movie_data[movie_data['title'].str.lower().str.contains(query)]
        
        # Convert to list of dictionaries
        movies = results.to_dict('records')
        
        return jsonify({"movies": movies})
    except Exception as e:
        logger.error(f"Error in search_movies: {str(e)}")
        return jsonify({"error": str(e)}), 500

# --- Log all registered routes for debugging and documentation ---
logger.info("--- Flask URL Map (Registered Routes) ---")
for rule in app.url_map.iter_rules():
    logger.info(f"Rule: {rule.endpoint} | Path: {rule.rule} | Methods: {rule.methods}")
logger.info("---------------------------------------")

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)