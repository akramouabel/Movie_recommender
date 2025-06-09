# Movie Recommendation System

A full-stack movie recommendation system with a Flask backend and modern frontend.

## Project Structure

```
movie_recommender/
├── app/                    # Main application package
│   ├── __init__.py        # Package initialization
│   ├── api/               # API routes and endpoints
│   │   ├── __init__.py
│   │   └── routes.py      # API route definitions
│   ├── core/              # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py      # Configuration settings
│   │   └── recommender.py # Recommendation engine
│   └── utils/             # Utility functions
│       ├── __init__.py
│       └── data_loader.py # Data loading utilities
├── data/                  # Data directory
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data files
├── scripts/              # Utility scripts
│   ├── generate_data.py  # Data generation script
│   └── setup.py         # Setup script
├── tests/               # Test directory
│   └── __init__.py
├── frontend/            # Frontend application
├── .gitignore
├── .gitattributes
├── Procfile            # Deployment configuration
├── requirements.txt    # Python dependencies
├── runtime.txt        # Python runtime version
└── wsgi.py           # WSGI entry point
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/movie_recommender.git
   cd movie_recommender
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Generate the required data:
   ```bash
   python scripts/generate_data.py
   ```

5. Run the development server:
   ```bash
   python wsgi.py
   ```

## Deployment

The application is configured for deployment on Render. The deployment process will:
1. Install dependencies from requirements.txt
2. Generate necessary data files
3. Start the Gunicorn server

## API Endpoints

- `GET /`: Health check endpoint
- `GET /recommend`: Get movie recommendations
- `GET /suggest`: Get movie title suggestions
- `GET /genres`: Get available movie genres
- `GET /years`: Get available movie years
- `GET /api/movies`: Get paginated movie list
- `GET /api/movies/<id>`: Get specific movie details
- `GET /api/recommendations/<id>`: Get recommendations for a movie
- `GET /api/search`: Search movies

## Development

To run the development server:
```bash
python wsgi.py
```

The server will start on http://localhost:5001

## Testing

To run tests:
```bash
python -m pytest tests/
```
