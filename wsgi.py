import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.main import app

if __name__ == "__main__":
    app.run() 