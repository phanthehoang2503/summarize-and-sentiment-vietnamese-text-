#!/usr/bin/env python3
"""
Project cleanup script
Removes temporary files, cache directories, and build artifacts
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """Remove temporary files and cache directories"""
    project_root = Path(__file__).parent
    
    # Directories to remove
    cache_dirs = [
        ".pytest_cache",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        "build",
        "dist",
        "*.egg-info"
    ]
    
    # File patterns to remove
    temp_files = [
        "*.pyc",
        "*.pyo", 
        "*.pyd",
        "*.tmp",
        "*.temp",
        "*.log"
    ]
    
    print("🧹 Cleaning up project...")
    
    # Remove cache directories
    for pattern in cache_dirs:
        for item in project_root.rglob(pattern):
            if item.is_dir():
                try:
                    shutil.rmtree(item)
                    print(f"  Removed directory: {item.relative_to(project_root)}")
                except Exception as e:
                    print(f"  Could not remove {item.relative_to(project_root)}: {e}")
    
    # Remove temporary files
    for pattern in temp_files:
        for item in project_root.rglob(pattern):
            if item.is_file():
                try:
                    item.unlink()
                    print(f"  Removed file: {item.relative_to(project_root)}")
                except Exception as e:
                    print(f"  Could not remove {item.relative_to(project_root)}: {e}")
    
    print("Cleanup completed!")

if __name__ == "__main__":
    cleanup_project()