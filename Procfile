release: |
  python -c "
  import os
  import sys
  import logging
  import time
  
  # Configure logging
  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(levelname)s - %(message)s'
  )
  
  # Check TMDB API key
  api_key = os.getenv('TMDB_API_KEY')
  if not api_key:
      logging.error('TMDB_API_KEY environment variable not set')
      sys.exit(1)
  logging.info('TMDB API key is present')
  
  # Set up paths
  os.environ['PYTHONPATH'] = os.getcwd()
  data_dir = '/opt/render/project/src/data'
  os.makedirs(data_dir, exist_ok=True)
  logging.info(f'Created/verified data directory: {data_dir}')
  
  # Run data generation
  try:
      from generate_data import *
      logging.info('Starting data generation...')
      __main__()
      logging.info('Data generation completed successfully')
      
      # Verify the file was created
      pkl_file = os.path.join(data_dir, 'movie_data_api.pkl')
      if os.path.exists(pkl_file):
          logging.info(f'Data file created successfully at: {pkl_file}')
      else:
          logging.error(f'Data file not found at: {pkl_file}')
          sys.exit(1)
  except Exception as e:
      logging.error(f'Error during data generation: {e}')
      sys.exit(1)
  "
web: gunicorn backend.main:app --bind 0.0.0.0:$PORT --timeout 120 