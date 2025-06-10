# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.

## Backend API Endpoints

This section details the RESTful API endpoints provided by the Flask backend for the Movie Recommendation System.

### Base URL: `https://movie-recommender-skvv.onrender.com`

---

### 1. Root Endpoint: `/`
- **Method:** `GET`
- **Purpose:** Health check to confirm the backend is running.
- **Response:** `{"message": "Movie Recommendation API is running!"}`

---

### 2. Movie Recommendations by Title: `/recommend`
- **Method:** `GET`
- **Purpose:** Provides a list of movie recommendations based on a given title, with optional filters.
- **Query Parameters:**
    - `title` (required): The movie title.
    - `top_n` (optional, default: 10): Number of recommendations.
    - `min_year` (optional): Filter by minimum release year.
    - `max_year` (optional): Filter by maximum release year.
    - `genres` (optional, JSON string): Filter by a list of genres (e.g., `["Action", "Sci-Fi"]`).
- **Response:** List of detailed movie objects, including `id`, `title`, `release_date` (year), `genres`, `overview`, `poster_path`.
- **Example:** `/recommend?title=The Dark Knight&top_n=5&genres=["Action", "Crime"]`

---

### 3. Movie Title Suggestions: `/suggest`
- **Method:** `GET`
- **Purpose:** Provides autocomplete suggestions for movie titles.
- **Query Parameter:** `query` (required): Partial movie title.
- **Response:** List of matching movie titles.
- **Example:** `/suggest?query=avat`

---

### 4. Get All Unique Genres: `/genres`
- **Method:** `GET`
- **Purpose:** Retrieves all unique movie genres available in the dataset.
- **Response:** List of sorted unique genres.

---

### 5. Get All Unique Years: `/years`
- **Method:** `GET`
- **Purpose:** Retrieves all unique movie release years available in the dataset.
- **Response:** List of sorted unique years.

---

### 6. Paginated List of Movies: `/api/movies`
- **Method:** `GET`
- **Purpose:** Fetches a paginated list of movies.
- **Query Parameters:**
    - `page` (optional, default: 1): Page number.
    - `per_page` (optional, default: 20): Items per page.
- **Response:** Paginated list of movie objects with total count and page info.
- **Example:** `/api/movies?page=1&per_page=10`

---

### 7. Comprehensive Movie Details by ID: `/api/movies/<int:movie_id>`
- **Method:** `GET`
- **Purpose:** Fetches all detailed information for a single movie by its ID.
- **URL Parameter:** `movie_id` (required): The unique integer ID of the movie.
- **Response:** Comprehensive JSON object with all available movie details.
- **Example:** `/api/movies/19995`

---

### 8. Movie Recommendations by ID: `/api/recommendations/<int:movie_id>`
- **Method:** `GET`
- **Purpose:** Provides movie recommendations based on a specific movie ID, using the same high-quality logic as `/recommend`.
- **URL Parameter:** `movie_id` (required): The unique integer ID of the movie.
- **Query Parameter:** `top_n` (optional, default: 10): Number of recommendations.
- **Response:** List of detailed movie objects, similar to `/recommend`.
- **Example:** `/api/recommendations/19995?top_n=5`

---

### 9. Search Movies by Title: `/api/search`
- **Method:** `GET`
- **Purpose:** Searches for movies by a partial or full title.
- **Query Parameter:** `query` (required): The search term.
- **Response:** List of matching movie objects.
- **Example:** `/api/search?query=spider`
