release: |
  python -c "
  import os
  import sys
  import logging
  logging.basicConfig(level=logging.INFO)
  if not os.getenv('TMDB_API_KEY'):
      logging.error('TMDB_API_KEY environment variable not set')
      sys.exit(1)
  os.environ['PYTHONPATH'] = os.getcwd()
  from generate_data import *
  __main__()
  "
web: gunicorn app:app 