"""
Path utilities for the project
"""
from pathlib import Path
import os

def get_project_root() -> Path:
    """
    Get the project root directory regardless of where this is called from
    
    Returns:
        Path: Project root directory
    """
    # Find the project root by looking for a marker file
    current = Path(__file__).resolve()
    
    # Go up the directory tree until we find the project root
    # Look for common markers like requirements.txt, README.md, or .git
    markers = ['requirements.txt', 'README.md', '.git', 'setup.py']
    
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    
    # Fallback: assume the project root is two levels up from this file
    return current.parent.parent

def ensure_path_exists(path: Path) -> Path:
    """
    Ensure a directory path exists, create if it doesn't
    
    Args:
        path: Path to ensure exists
        
    Returns:
        Path: The path (now guaranteed to exist)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_data_path(filename: str, subfolder: str = "raw") -> Path:
    """
    Get absolute path to a data file
    
    Args:
        filename: Name of the file
        subfolder: Subfolder in data directory (raw, processed, etc.)
        
    Returns:
        Path: Absolute path to the file
    """
    return get_project_root() / "data" / subfolder / filename

def get_model_path(model_name: str = "checkpoint-1854") -> Path:
    """
    Get absolute path to a model directory
    
    Args:
        model_name: Name of the model checkpoint
        
    Returns:
        Path: Absolute path to the model
    """
    return get_project_root() / "models" / "summarizer" / model_name

def get_cache_path() -> Path:
    """
    Get absolute path to cache directory
    
    Returns:
        Path: Absolute path to cache
    """
    return get_project_root() / "cache"

# Example usage functions
if __name__ == "__main__":
    pass
