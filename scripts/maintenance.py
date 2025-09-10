#!/usr/bin/env python3
"""
Simple maintenance script for regular cleanup
"""
import sys
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).parent.parent

def quick_cleanup():
    """Quick cleanup of common temporary files"""
    print("Running quick cleanup...")
    
    # Remove __pycache__
    for pycache in PROJECT_ROOT.rglob('__pycache__'):
        if pycache.is_dir():
            shutil.rmtree(pycache)
            print(f"Removed {pycache}")
    
    # Remove .pyc files
    for pyc_file in PROJECT_ROOT.rglob('*.pyc'):
        pyc_file.unlink()
        print(f"Removed {pyc_file}")
    
    print("Quick cleanup completed!")

if __name__ == "__main__":
    quick_cleanup()
