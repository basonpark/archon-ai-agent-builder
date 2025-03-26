"""
Environment variable loader utility.

This module helps all iteration versions load environment variables from the root .env file.
Each version directory can simply import this module instead of creating its own .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

def load_root_env():
    """
    Load environment variables from the .env file in the project root.
    
    This function finds the project root directory (parent of the iterations folder)
    and loads the .env file from there, regardless of where in the project the 
    function is called from.
    
    Returns:
        bool: True if the .env file was found and loaded, False otherwise
    """
    # Get the current file's directory (should be inside iterations/)
    current_dir = Path(__file__).parent.absolute()
    
    # Get the project root directory (parent of iterations/)
    root_dir = current_dir.parent
    
    # Path to the .env file
    env_path = root_dir / '.env'
    
    if env_path.exists():
        # Load the environment variables
        load_dotenv(dotenv_path=env_path)
        return True
    else:
        print(f"Warning: .env file not found at {env_path}")
        return False
        
# Load environment variables immediately when this module is imported
loaded = load_root_env() 