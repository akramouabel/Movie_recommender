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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask Application Initialization ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Global variables for data
movies_df = None
similarity_matrix = None

def load_data():
    """Load the movie data and similarity matrix from the pickle file."""
    global movies_df, similarity_matrix
    
    # Try to load from the Render deployment path first
    pkl_path = '/opt/render/project/src/data/movie_data_api.pkl'
    if not os.path.exists(pkl_path):
        # Fall back to local path
        pkl_path = 'data/movie_data_api.pkl'
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            movies_df = data['movies_df']
            similarity_matrix = data['similarity_matrix']
        logging.info(f"Successfully loaded data from {pkl_path}")
        return True
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return False

# Load data when the app starts
if not load_data():
    logging.critical("CRITICAL: Recommendation data could not be loaded on backend startup. Flask app may not function correctly.")

# --- API Routes ---

@app.route('/')
def home():
    """
    Root endpoint to confirm the backend is running.
    Returns a simple status message.
    """
    return jsonify({"message": "Movie Recommendation API is running!"})

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Get movie recommendations based on a movie title."""
    try:
        data = request.get_json()
        movie_title = data.get('title', '').strip()
        
        if not movie_title:
            return jsonify({"error": "No movie title provided"}), 400
            
        # Find the movie index
        movie_idx = movies_df[movies_df['title'].str.lower() == movie_title.lower()].index
        if len(movie_idx) == 0:
            return jsonify({"error": f"Movie '{movie_title}' not found"}), 404
            
        movie_idx = movie_idx[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(similarity_matrix[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Get top 10 similar movies
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Get movie details
        recommendations = movies_df.iloc[movie_indices][['movie_id', 'title', 'poster_path', 'vote_average', 'release_date']].to_dict('records')
        
        return jsonify({"recommendations": recommendations})
        
    except Exception as e:
        logging.error(f"Error in recommend endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/suggest', methods=['GET'])
def suggest():
    """Get movie suggestions based on a search query."""
    try:
        query = request.args.get('q', '').strip().lower()
        if not query:
            return jsonify({"suggestions": []})
            
        # Filter movies based on the query
        matches = movies_df[movies_df['title'].str.lower().str.contains(query)]
        
        # Get top 5 matches
        suggestions = matches.head(5)[['movie_id', 'title', 'poster_path']].to_dict('records')
        
        return jsonify({"suggestions": suggestions})
        
    except Exception as e:
        logging.error(f"Error in suggest endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/genres', methods=['GET'])
def get_genres():
    """Get list of all unique genres."""
    try:
        # Get all unique genres
        all_genres = set()
        for genres in movies_df['genres']:
            all_genres.update(genres)
            
        return jsonify({"genres": sorted(list(all_genres))})
        
    except Exception as e:
        logging.error(f"Error in genres endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/years', methods=['GET'])
def get_years():
    """Get list of all unique release years."""
    try:
        # Extract years from release dates
        years = movies_df['release_date'].str[:4].unique()
        years = sorted([int(y) for y in years if y.isdigit()])
        
        return jsonify({"years": years})
        
    except Exception as e:
        logging.error(f"Error in years endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001))) 