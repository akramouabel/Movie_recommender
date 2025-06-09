# Movie Recommendation System Backend

This project provides a robust Flask-based backend for a movie recommendation system. It leverages movie metadata, textual overviews, genres, and release years to offer relevant movie suggestions. The system is designed for easy deployment and efficient data handling.

## Features

*   **Content-Based Recommendations:** Generates movie recommendations based on similarity to a given movie title.
*   **Fuzzy Title Matching:** Accurately matches user input to existing movie titles in the database.
*   **Genre and Year Filtering:** Allows users to filter recommendations by genre and release year.
*   **Pagination for Movie Listings:** Provides paginated access to the entire movie catalog.
*   **Movie Details Retrieval:** Fetches detailed information for a specific movie by its ID.
*   **Dynamic Genre and Year Lists:** Endpoints to retrieve all unique genres and release years for frontend filtering.
*   **Optimized Data Loading:** Pre-computes and loads large data components (embeddings, clusters) during the build phase for fast application startup.
*   **Docker-Ready:** Designed for containerized deployment, specifically tested with Render.

## Project Structure

```
.
├── backend/
│   ├── __init__.py
│   ├── main.py             # Flask application entry point and API routes
│   └── recommender.py      # Core recommendation logic, data loading, similarity
├── data/                   # Directory for generated data files (created during build)
│   ├── movie_data_api.pkl
│   ├── movie_embeddings.npy
│   └── movie_clusters.npy
├── generate_data.py        # Script to fetch data, generate embeddings, and clusters
├── run_backend.py          # Script to run the Flask backend locally
└── requirements.txt        # Python dependencies
└── README.md               # This documentation file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your_repository_url>
    cd movie-recommendation-system
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    ```
    *   On Windows: `.venv\Scripts\activate`
    *   On macOS/Linux: `source .venv/bin/activate`
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up TMDB API Key:**
    Obtain an API key from [The Movie Database (TMDB) API](https://www.themoviedb.org/documentation/api).
    Set it as an environment variable named `TMDB_API_KEY`.
    *   On Windows (for current session): `set TMDB_API_KEY=YOUR_API_KEY`
    *   On macOS/Linux (for current session): `export TMDB_API_KEY=YOUR_API_KEY`
    *   For persistent setup, consider adding it to your shell profile (`.bashrc`, `.zshrc`, etc.) or using a `.env` file and a library like `python-dotenv`.

## Data Generation

The `generate_data.py` script fetches movie data from the TMDB API, processes it, generates text embeddings for similarity calculations, and performs K-Means clustering for diversity in recommendations.

To generate the data locally:

```bash
python generate_data.py
```
This process can take several minutes depending on the number of movies fetched and your system's resources. The generated files (`movie_data_api.pkl`, `movie_embeddings.npy`, `movie_clusters.npy`) will be saved in the `data/` directory.

## Running the Backend Locally

Once the data is generated, you can run the Flask backend:

```bash
python run_backend.py
```
The server will typically run on `http://127.0.0.1:5001`.

## Deployment on Render

This project is configured for deployment on Render.com. Follow these steps to deploy your web service:

1.  **Create a new Web Service on Render:** Connect your GitHub repository.
2.  **Configure Build Command:**
    In your Render service settings, set the **Build Command** to:
    ```bash
    pip install -r requirements.txt && python generate_data.py
    ```
    This command ensures that all dependencies are installed and the necessary data files are generated *before* your application starts.
3.  **Configure Start Command:**
    Set the **Start Command** to:
    ```bash
    gunicorn backend.main:app --bind 0.0.0.0:$PORT --timeout 120
    ```
    This tells Gunicorn to run your Flask application (`app` instance in `backend.main` module) and bind it to the port provided by Render's environment variable.
4.  **Set Environment Variable:**
    Add a new environment variable `TMDB_API_KEY` with your actual TMDB API key in Render's environment settings.
5.  **Adjust Health Check Timeout (Crucial for larger datasets):**
    In your Render service settings, under the Health Check section, increase the **Timeout** value. For larger datasets (e.g., 2000-3000 movies), a value of `300` seconds (5 minutes) is recommended to allow the application sufficient time to load data during startup. Render has a fixed 15-minute deployment timeout, which this helps to avoid.

## API Endpoints

The backend exposes the following API endpoints:

*   **`/` (GET):**
    *   Returns a simple status message to confirm the backend is running.
    *   Example: `GET /`
    *   Response: `{"message": "Movie Recommendation API is running!"}`

*   **`/recommend` (GET):**
    *   **Parameters:** `title` (string, required), `top_n` (int, optional, default: 10), `min_year` (int, optional), `max_year` (int, optional), `genres` (JSON array of strings, optional)
    *   Returns a list of recommended movies based on the provided title and filters.
    *   Example: `GET /recommend?title=Inception&top_n=5`
    *   Example: `GET /recommend?title=Avatar&min_year=2000&max_year=2010&genres=["Action", "Adventure"]`

*   **`/suggest` (GET):**
    *   **Parameters:** `query` (string, required)
    *   Returns a list of movie title suggestions based on a partial query (for autocomplete).
    *   Example: `GET /suggest?query=inc`

*   **`/genres` (GET):**
    *   Returns a list of all unique movie genres available in the dataset.
    *   Example: `GET /genres`

*   **`/years` (GET):**
    *   Returns a list of all unique movie release years available in the dataset.
    *   Example: `GET /years`

*   **`/api/movies` (GET):**
    *   **Parameters:** `page` (int, optional, default: 1), `per_page` (int, optional, default: 20)
    *   Returns a paginated list of all movies in the dataset.
    *   Example: `GET /api/movies?page=1&per_page=10`

*   **`/api/movies/<int:movie_id>` (GET):**
    *   **Parameters:** `movie_id` (int, required, path parameter)
    *   Returns detailed information for a specific movie by its ID.
    *   Example: `GET /api/movies/10138`

*   **`/api/recommendations/<int:movie_id>` (GET):**
    *   **Parameters:** `movie_id` (int, required, path parameter)
    *   Returns a list of recommendations based on a specific movie ID (similar to `/recommend` but using ID).
    *   Example: `GET /api/recommendations/10138`

*   **`/api/search` (GET):**
    *   **Parameters:** `query` (string, required)
    *   Performs a basic search for movies by title containing the query string.
    *   Example: `GET /api/search?query=iron`

## Key Improvements and Fixes Made During Development

This project underwent significant refinement to ensure robust deployment and functionality:

1.  **Initial 500 Error (`Movie 'Inception' not found`):**
    *   **Problem:** The backend was unable to find "Inception" initially.
    *   **Resolution:** Discovered `movie_data_api.pkl` was loaded as a tuple, not a dictionary, leading to incorrect DataFrame extraction. `backend/main.py` and `backend/recommender.py` were updated to handle both tuple and dictionary formats for the loaded pickle data.

2.  **Deployment Timeout on Render (502/Timed Out):**
    *   **Problem:** The service was failing to start within Render's deployment timeout.
    *   **Resolution:**
        *   The Flask application's port binding was corrected in `run_backend.py` to correctly use Render's `$PORT` environment variable.
        *   The Gunicorn start command on Render was updated to `gunicorn backend.main:app --bind 0.0.0.0:$PORT --timeout 120`.
        *   Identified that `SentenceTransformer` model loading and K-Means clustering were happening at runtime, consuming excessive resources. `generate_data.py` was modified to pre-compute and save `movie_clusters.npy` during the build phase.
        *   Unnecessary `SentenceTransformer` and `torch` imports were removed from `backend/recommender.py` to prevent large model loading during runtime.
        *   **Crucial:** The Render web service's health check timeout was recommended to be increased to 300 seconds (5 minutes) to accommodate initial data loading.

3.  **`KeyError: 'id'` during Data Loading:**
    *   **Problem:** After fixing the `movie_data_api.pkl` format, `backend/main.py` still had an issue extracting the DataFrame columns.
    *   **Resolution:** Corrected `backend/main.py` to directly assign `data['movies_df']` to `movie_data` when the loaded object was a dictionary.

4.  **`KeyError: 'movie_id'` during Recommendation:**
    *   **Problem:** The `get_recommendations` function in `backend/recommender.py` was trying to access a non-existent column named `'movie_id'` instead of the correct `'id'`.
    *   **Resolution:** All instances of `'movie_id'` were replaced with `'id'` in `backend/recommender.py` where DataFrame columns were accessed.

5.  **Dataset Size Management for Free Tier:**
    *   **Problem:** Attempting to generate the full 4800-movie dataset caused resource issues on Render's free tier, leading to `movie_clusters.npy` not being generated.
    *   **Resolution:** The number of pages fetched in `generate_data.py` was adjusted to 100 pages (~2000 movies), allowing the build and data generation process to complete successfully within the free tier's resource limits. Later, this was incrementally increased to 150 pages (~3000 movies) as a next optimization step.

This comprehensive `README.md` details the project's features, setup, and the solutions to the various challenges encountered, providing a clear guide for anyone using or contributing to the project.
