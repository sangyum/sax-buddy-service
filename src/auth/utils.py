"""Authentication utility functions"""

import os


def is_development_mode() -> bool:
    """Check if application is running in development mode
    
    Returns:
        True if ENVIRONMENT=development, False otherwise
    """
    return os.getenv("ENVIRONMENT", "production").lower() == "development"