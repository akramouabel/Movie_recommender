release: python -c "import os; os.environ['TMDB_API_KEY'] = '$TMDB_API_KEY'; from generate_data import *; __main__()"
web: gunicorn app:app 