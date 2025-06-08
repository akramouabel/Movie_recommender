# backend/main.py
# Flask Backend for Movie Recommendation System
# ---------------------------------------------
# This file defines the main API endpoints for the movie recommender backend.
# It handles requests from the frontend, applies filters, and returns recommendations or metadata.

from flask import Flask, request, jsonify
import os
import logging
import pandas as pd
import json # Explicitly import json for parsing genres
from backend import recommender # Your recommender module
from flask_cors import CORS

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask Application Initialization ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- API Routes ---

@app.route('/')
def home():
    """
    Root endpoint to confirm the backend is running.
    Returns a simple status message.
    """
    return "Movie Recommendation Backend is running!"

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
    # Get the genres parameter as a raw string from the request (it's JSON-dumped by frontend)
    selected_genres_raw_str = request.args.get('genres') 

    selected_genres_list = [] # Initialize as an empty list
    if selected_genres_raw_str:
        try:
            # Attempt to parse the JSON string into a Python list
            temp_genres = json.loads(selected_genres_raw_str)
            # Ensure the result is a list and then clean its contents
            if isinstance(temp_genres, list):
                selected_genres_list = [g.strip().replace(" ", "") for g in temp_genres if isinstance(g, str) and g.strip()]
            else:
                logging.warning(f"Expected list for genres after JSON decode but got {type(temp_genres)}: {temp_genres}")
                # Fallback in case of unexpected format
                selected_genres_list = [g.strip().replace(" ", "") for g in selected_genres_raw_str.split(',') if g.strip()]
        except json.JSONDecodeError:
            logging.error(f"Failed to decode genres JSON: {selected_genres_raw_str}. Treating as no genres selected.")
            selected_genres_list = [] # Treat as no genres selected if JSON parsing fails
        except Exception as e:
            logging.error(f"Unexpected error processing genres: {e}")
            selected_genres_list = [] # Fallback for other errors
    logging.info(f"Recommendation request for title: '{movie_title}', min_year: {min_year}, max_year: {max_year}, genres: {selected_genres_list}")

    if not movie_title:
        return jsonify({"error": "Movie title parameter 'title' is required."}), 400

    # Pass the already parsed Python list directly to recommender.get_recommendations
    recommendations, matched_title = recommender.get_recommendations(
        movie_title=movie_title,
        num_recommendations=top_n,
        min_year=min_year,
        max_year=max_year,
        selected_genres=selected_genres_list # This is now a Python list
    )

    if recommendations is None:
        logging.error(f"Recommender returned None for '{movie_title}': {matched_title}")
        return jsonify({"error": matched_title}), 500
    elif not recommendations:
        logging.info(f"No recommendations found for '{movie_title}' with applied filters. Message: {matched_title}")
        return jsonify({"recommendations": [], "message": matched_title}), 200
    else:
        logging.info(f"Successfully retrieved {len(recommendations)} recommendations for '{movie_title}'.")
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
    logging.info(f"Suggestions for '{query}': {suggestions}")
    return jsonify({"suggestions": suggestions}), 200

@app.route('/genres', methods=['GET'])
def genres():
    """
    Returns a list of all unique movie genres available in the dataset.
    Used to populate filter options in the frontend.
    """
    if recommender.movies_df is None:
        logging.error("movies_df not loaded in recommender module when /genres was requested. Data pre-loading failed.")
        return jsonify({"error": "Movie data not loaded on backend. Please ensure data generation script was run."}), 500
    all_genres = []
    for movie_genres in recommender.movies_df['genres'].dropna():
        if isinstance(movie_genres, list):
            all_genres.extend(movie_genres)
    unique_genres = sorted(list(set(all_genres)))
    logging.info(f"Fetched {len(unique_genres)} unique genres.")
    return jsonify({"genres": unique_genres})

@app.route('/years', methods=['GET'])
def years():
    """
    Returns a list of all unique movie release years available in the dataset.
    Used to populate filter options in the frontend.
    """
    if recommender.movies_df is None:
        logging.error("movies_df not loaded in recommender module when /years was requested. Data pre-loading failed.")
        return jsonify({"error": "Movie data not loaded on backend. Please ensure data generation script was run."}), 500
    years_list = []
    for date_str in recommender.movies_df['release_date'].dropna():
        converted_date = pd.to_datetime(date_str, errors='coerce')
        if pd.notna(converted_date): 
            years_list.append(converted_date.year)
    unique_years = sorted(list(set(years_list)))
    logging.info(f"Fetched {len(unique_years)} unique years.")
    return jsonify({"years": unique_years})

# --- Log all registered routes for debugging and documentation ---
logging.info("--- Flask URL Map (Registered Routes) ---")
for rule in app.url_map.iter_rules():
    logging.info(f"Rule: {rule.endpoint} | Path: {rule.rule} | Methods: {rule.methods}")
logging.info("---------------------------------------")

if __name__ == '__main__':
    # Initial data load for recommender system when Flask app starts
    if not recommender.load_recommendation_data():
        logging.critical("CRITICAL ERROR: Failed to load recommendation data. Exiting Flask app.")
        exit(1) # Exit if data loading fails, as the app cannot function without it
    port = int(os.environ.get('PORT', 5000))  # Use Render's port if set, else 5000
    app.run(host='0.0.0.0', port=port, debug=True)