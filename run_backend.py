# run_backend.py
# Script to launch the Flask backend for the Movie Recommendation System
# ---------------------------------------------------------------------
# This script sets up the environment and runs the Flask development server.
# It ensures the correct working directory and environment variables are set for Flask.

import os
import subprocess
import sys
import logging # Import logging for more structured output if needed

# Configure basic logging for the runner script itself
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    # This ensures the code inside this block only runs when the script is executed directly.
    # It prevents accidental execution if this file were to be imported as a module elsewhere.

    logging.info("--- Starting Flask Backend ---")

    # --- Directory Management ---
    # Get the directory where this script (run_backend.py) is located.
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change the current working directory to the project root.
    # This is crucial because Flask expects 'FLASK_APP' (e.g., 'backend.main')
    # to be discoverable from the current working directory. If you run
    # `python run_backend.py` from a different folder, Flask might not find 'backend'.
    os.chdir(current_script_dir) 
    logging.info(f"Changed working directory to: {os.getcwd()}")

    # --- Environment Variable Configuration for Flask ---
    # FLASK_APP: Tells the 'flask' command where to find your Flask application instance.
    # 'backend.main' refers to the 'main.py' file inside the 'backend' package.
    # Flask will look for an 'app' (or 'application') variable within that module.
    os.environ['FLASK_APP'] = 'backend.main' 

    # FLASK_DEBUG: Enables Flask's debug mode.
    # In debug mode:
    # 1. The server will automatically reload when code changes are detected.
    # 2. An interactive debugger is activated for uncaught exceptions, useful for development.
    # IMPORTANT: 'FLASK_DEBUG' should generally be set to '0' (or completely unset) in production
    # due to security implications (e.g., exposing sensitive information via the debugger).
    os.environ['FLASK_DEBUG'] = '1' 

    # --- Constructing the Flask Command ---
    # sys.executable: Points to the Python interpreter currently being used.
    # This ensures the Flask command is run with the same Python environment
    # (including your virtual environment) that executed 'run_backend.py'.
    # '-m flask': Invokes the 'flask' module as a script. This is the standard way
    # to run Flask applications.
    # 'run': The Flask subcommand to start the development server.
    # '--host 0.0.0.0': Makes the Flask server accessible from any IP address on the network.
    #                  For local development, '127.0.0.1' (localhost) is common, but '0.0.0.0'
    #                  is necessary if you want to access it from other devices on your LAN
    #                  or from a deployed environment.
    # '--port 5001': Specifies the port on which the Flask server will listen.
    #                Ensures consistency with the frontend's API calls.
    flask_cmd = [sys.executable, '-m', 'flask', 'run', '--host', '0.0.0.0', '--port', '5001']

    logging.info(f"Executing Flask command: {' '.join(flask_cmd)}")

    # --- Running the Flask Process ---
    try:
        # subprocess.run: Executes the Flask command.
        # It waits for the command to complete. In this case, the Flask server
        # will run continuously until explicitly stopped (e.g., via Ctrl+C).
        # Any output from the Flask server will be directed to this terminal.
        subprocess.run(flask_cmd)
    except KeyboardInterrupt:
        # Catches the common Ctrl+C signal, allowing for a graceful shutdown message.
        logging.info("\nFlask backend stopped by user (KeyboardInterrupt).")
    except FileNotFoundError:
        # Specifically handles the case where 'flask' or 'python' command might not be found.
        logging.error(f"Error: Command not found. Ensure Python and Flask are installed and in your PATH.")
        logging.error(f"Attempted command: {' '.join(flask_cmd)}")
    except Exception as e:
        # Catches any other unexpected exceptions during the subprocess execution.
        logging.critical(f"An unexpected error occurred while starting Flask backend: {e}", exc_info=True)
        logging.error("Please ensure you have Flask installed (`pip install Flask`) and your backend files are correct.")