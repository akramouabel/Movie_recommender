# frontend/app.py
# Streamlit Frontend for Movie Recommendation System
# --------------------------------------------------
# This app provides a user-friendly interface to interact with the movie recommender backend.
# Users can search for movies, apply filters, and view AI-powered recommendations with rich details.

import streamlit as st
import requests
import json
import logging

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Backend Configuration ---
FLASK_BACKEND_URL = "http://127.0.0.1:5001"  # URL of the Flask backend API

# --- Streamlit UI Configuration ---
st.set_page_config(layout="wide", page_title="🎬 Movie Recommender", page_icon="🎬")

# --- Helper Function to Fetch Data from Backend ---
@st.cache_data(ttl=3600)  # Cache API calls for 1 hour to reduce backend load
def fetch_from_backend(endpoint, params=None):
    """
    Generic helper to fetch data from the backend API with error handling and logging.
    """
    try:
        url = f"{FLASK_BACKEND_URL}/{endpoint}"
        logging.info(f"Fetching from: {url} with params: {params}")
        response = requests.get(url, params=params, timeout=15)  # 15-second timeout
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json()
    except requests.exceptions.Timeout:
        st.error(f"Request to backend timed out. Please try again. ({endpoint})")
        logging.error(f"Timeout fetching from {endpoint}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend server. Please ensure the Flask backend is running.")
        logging.error(f"Connection error fetching from {endpoint}")
        return None
    except requests.exceptions.HTTPError as e:
        error_msg_display = f"Backend error: {e.response.status_code} - {e.response.text}"
        st.error(error_msg_display)
        logging.error(f"HTTP error fetching from {endpoint}: {e.response.status_code} - {e.response.text}")
        return None
    except json.JSONDecodeError:
        st.error("Failed to decode JSON from backend. Malformed response.")
        if 'response' in locals() and response.text:
            logging.error(f"JSON decode error from {endpoint}: {response.text}")
        else:
            logging.error(f"JSON decode error from {endpoint}: No response text available.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logging.error(f"Unexpected error fetching from {endpoint}: {e}")
        return None

# --- Main Streamlit App Layout ---
st.title("🎬 Movie Recommender")
st.write("Find movies you'll love! Powered by advanced AI recommendations.")

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Fetch genres and years from backend for filter widgets
@st.cache_data
def fetch_genres():
    """Fetches the list of available genres from the backend."""
    resp = requests.get("http://localhost:5001/genres")
    return resp.json().get("genres", [])

@st.cache_data
def fetch_years():
    """Fetches the list of available release years from the backend."""
    resp = requests.get("http://localhost:5001/years")
    return resp.json().get("years", [])

genres_list = fetch_genres()
years_list = fetch_years()
if years_list:
    min_year, max_year = min(years_list), max(years_list)
else:
    min_year, max_year = 1980, 2023  # Fallback defaults

# Sidebar filter widgets for user customization
selected_genres = st.sidebar.multiselect("Genres", genres_list)
year_range = st.sidebar.slider("Release Year", min_year, max_year, (2000, 2023))
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 10.0, 6.0)

# --- Search Bar ---
# Main input for the user to enter a movie title
movie_query = st.text_input("Enter a movie title")

# --- Fetch Recommendations ---
def get_recommendations(title, genres, year_range, min_rating):
    """
    Fetches movie recommendations from the backend based on user input and filters.
    Filters results by minimum rating on the frontend.
    """
    params = {
        "title": title,
        "top_n": 9,  # Request exactly 9 recommendations from backend
        "min_year": year_range[0],
        "max_year": year_range[1],
        "genres": str(genres)
    }
    resp = requests.get("http://localhost:5001/recommend", params=params)
    data = resp.json()
    recs = data.get("recommendations", [])
    # Filter by min_rating on frontend if needed
    return [r for r in recs if r.get("vote_average", 0) >= min_rating]

# Base URL for movie poster images (TMDB or similar)
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# --- Main Recommendation Display ---
if movie_query:
    with st.spinner("Loading recommendations..."):
        recommendations = get_recommendations(movie_query, selected_genres, year_range, min_rating)
    if recommendations:
        # Display recommendations in a 3-column grid
        cols = st.columns(3)
        for idx, movie in enumerate(recommendations):
            with cols[idx % 3]:
                # Handle poster image URL or fallback
                poster_path = movie.get('poster_path')
                if poster_path and poster_path.startswith('/'):
                    image_url = IMAGE_BASE_URL + poster_path
                elif poster_path and poster_path.startswith('http'):
                    image_url = poster_path
                else:
                    image_url = "https://via.placeholder.com/180x270?text=No+Image"
                st.image(image_url, width=180)
                # Movie title and year
                st.markdown(f"**{movie['title']}** ({movie['release_date'][:4]})")
                # Genres and rating
                st.markdown(f"Genres: {', '.join(movie['genres'])}")
                st.markdown(f"⭐ {movie['vote_average']} ({movie['vote_count']} votes)")
                # Expandable overview for better UX
                with st.expander("Overview"):
                    st.write(movie['overview'])
                st.markdown("---")
    else:
        st.error("No recommendations found. Try a different movie or filters.")
# End of file