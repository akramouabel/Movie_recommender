release: |
  #!/bin/bash
  echo "Starting data generation process..."
  
  # Check TMDB API key
  if [ -z "$TMDB_API_KEY" ]; then
    echo "ERROR: TMDB_API_KEY environment variable not set"
    exit 1
  fi
  echo "TMDB API key is present"
  
  # Set up paths
  export PYTHONPATH=$PWD
  DATA_DIR="/opt/render/project/src/data"
  mkdir -p $DATA_DIR
  echo "Created/verified data directory: $DATA_DIR"
  
  # Run data generation
  echo "Running data generation script..."
  python generate_data.py
  
  # Check if data file was created
  if [ -f "$DATA_DIR/movie_data_api.pkl" ]; then
    echo "Data file created successfully at: $DATA_DIR/movie_data_api.pkl"
    ls -l "$DATA_DIR/movie_data_api.pkl"
  else
    echo "ERROR: Data file not found at: $DATA_DIR/movie_data_api.pkl"
    exit 1
  fi

web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 