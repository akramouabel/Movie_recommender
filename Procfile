release: bash -c "
  echo 'RENDER_RELEASE_START: Starting data generation process'
  if [ -z \"$TMDB_API_KEY\" ]; then
    echo \"RENDER_RELEASE_ERROR: TMDB_API_KEY environment variable not set\"
    exit 1
  fi
  echo 'RENDER_RELEASE_INFO: TMDB API key is present'

  export PYTHONPATH=$PWD
  DATA_DIR=\"/opt/render/project/src/data\"
  mkdir -p $DATA_DIR || { echo 'RENDER_RELEASE_ERROR: Failed to create data directory'; exit 1; }
  echo \"RENDER_RELEASE_INFO: Created/verified data directory: $DATA_DIR\"

  echo 'RENDER_RELEASE_INFO: Running generate_data.py script'
  python generate_data.py || { echo 'RENDER_RELEASE_ERROR: generate_data.py script failed to execute'; exit 1; }
  echo 'RENDER_RELEASE_INFO: generate_data.py script finished execution'

  if [ -f \"$DATA_DIR/movie_data_api.pkl\" ]; then
    echo \"RENDER_RELEASE_SUCCESS: Data file created successfully at: $DATA_DIR/movie_data_api.pkl\"
    ls -l \"$DATA_DIR/movie_data_api.pkl\"
  else
    echo \"RENDER_RELEASE_CRITICAL: Data file NOT FOUND at: $DATA_DIR/movie_data_api.pkl\"
    exit 1
  fi
  echo 'RENDER_RELEASE_END: Release phase completed'
"
web: python generate_data.py && gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 