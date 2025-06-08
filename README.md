# 🎬 Intelligent Movie Recommender System

## Project Overview

This project is an intelligent movie recommendation system featuring a **Flask-based backend API** and a **Streamlit-powered web frontend**. Users can get personalized movie recommendations by entering a movie title, with options to filter results by genre, release year, and minimum rating. The system leverages advanced content-based filtering using movie overviews, genres, and embeddings for high-quality recommendations.

---

> **⚠️ Note: This repository uses [Git LFS](https://git-lfs.github.com/) for large files (such as `.pkl` and `.npy` data files).**
> 
> Before cloning or pulling, please install Git LFS:
> - [Download Git LFS](https://git-lfs.github.com/) and run `git lfs install` (once per machine).
> - After installing, you can clone and pull as usual. Git LFS will handle large files automatically.
> - If you do not install Git LFS, you will only get pointer files instead of the actual data.

---

## Features

- **Personalized Recommendations:** Get a list of movies similar to a given title.
- **Fuzzy Title Matching:** Smartly matches user input to actual movie titles in the database.
- **Genre & Year Filtering:** Filter recommendations by one or more genres and by release year range.
- **Minimum Rating Filter:** Only see movies above a certain rating.
- **Rich Movie Details:** See posters, release year, IMDb rating, genres, and an expandable overview for each recommendation.
- **Modern UI:** Responsive, user-friendly interface with instant feedback.
- **Modular Architecture:** Clean separation between backend (API) and frontend (UI) for easy maintenance and future expansion.

---

## System Architecture

**1. Backend (Flask API)**
- Loads pre-processed movie data and embeddings for fast, scalable recommendations.
- Exposes endpoints for recommendations, genre/year lists, and title suggestions.
- Handles all filtering, fuzzy matching, and similarity calculations.

**2. Frontend (Streamlit App)**
- Provides an interactive web interface for users.
- Communicates with the backend via HTTP requests.
- Displays recommendations in a visually appealing, filterable grid.

**3. Data Pipeline**
- Raw data (from TMDB CSVs) is processed by `generate_data.py` to extract features, compute embeddings, and save everything in a compact `.pkl` file for the backend.

---

## Data Flow

1. **Data Generation:**  
   Run `generate_data.py` to process raw CSVs and create `data/movie_data_api.pkl` and embeddings.  
   _This step is required only once, unless you want to update the dataset._

2. **Backend Startup:**  
   The Flask backend loads the processed data and embeddings into memory for fast API responses.

3. **Frontend Startup:**  
   The Streamlit app loads, fetches available genres/years, and waits for user input.

4. **User Interaction:**  
   The user enters a movie title and selects filters.

5. **Recommendation Request:**  
   The frontend sends a request to the backend, which returns a list of recommended movies.

6. **Display:**  
   The frontend displays the recommendations, including posters, genres, ratings, and overviews.

---

## Technologies Used

- **Python 3.8+**
- **Flask** (backend API)
- **Streamlit** (frontend UI)
- **Pandas, NumPy** (data processing)
- **Scikit-learn** (clustering, similarity)
- **Sentence-Transformers** (embeddings)
- **Requests** (HTTP requests)
- **Flask-CORS** (cross-origin support)
- **difflib** (fuzzy matching)

---

## Getting Started: Setup & Running Locally

### 1. Prerequisites

- Python 3.8 or higher
- `pip` (Python package installer)
- (Recommended) [virtualenv](https://virtualenv.pypa.io/en/latest/) for isolated environments

### 2. Unzip the Project

- Download the provided ZIP file and extract it to your desired location (e.g., `C:/Users/yourname/Documents/new_sys_version_3`).
- Open a terminal or command prompt and navigate to the extracted project folder:

```bash
cd path/to/new_sys_version_3
```

### 3. Create and Activate a Virtual Environment (Recommended)

**On Windows:**

```bash
python -m venv venv
./venv/Scripts/activate
```

**On macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

With the virtual environment activated, install all required packages:

```bash
pip install -r requirements.txt
```

### 5. Prepare the Data

- Place the raw TMDB CSV files (`tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`) in the `raw_data/` directory (create it if it doesn't exist).
- Run the data generation script to process the raw data:

```bash
python generate_data.py
```

This will create the processed data and embeddings in the `data/` directory.

### 6. Start the Backend

In the same terminal (with the virtual environment still activated):

```bash
python run_backend.py
```

- The backend will start on `http://127.0.0.1:5001` by default.

### 7. Start the Frontend

Open a **new terminal** (and activate the virtual environment again if needed):

**On Windows:**
```bash
./venv/Scripts/activate
```
**On macOS/Linux:**
```bash
source venv/bin/activate
```

Then run:
```bash
streamlit run frontend/app.py
```

- The frontend will open in your browser (usually at `http://localhost:8501`).

---

## Usage

- Enter a movie title in the search bar.
- Optionally, select genres, year range, and minimum rating in the sidebar.
- Recommendations will appear instantly as you type.

---

## Contributing & Documentation

- Please document any new features or changes in the codebase.
- For detailed API documentation, see the docstrings in `backend/main.py` and `backend/recommender.py`.
- For UI/UX changes, see `frontend/app.py`.

---

## Troubleshooting

- If you see errors about missing data, make sure you have run `generate_data.py` and have the required files in `raw_data/`.
- If you have issues with dependencies, try creating a new virtual environment and reinstalling requirements.
