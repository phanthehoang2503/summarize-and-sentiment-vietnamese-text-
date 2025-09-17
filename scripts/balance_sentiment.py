#!/usr/bin/env python3
"""
Script to balance the sentiment dataset
"""
import sys
from pathlib import Path

# Add project root to path for legacy compatibility
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import preprocess_sentiment
from app.core.config import get_config
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Get configuration
config = get_config()

def main():
    """
    Main function to reprocess sentiment data with balancing
    """
    raw_path = str(config.raw_data_dir)
    processed_path = str(config.processed_data_dir)
    
    print("=" * 60)
    print("SENTIMENT DATASET BALANCING")
    print("=" * 60)
    
    print(f"Raw data path: {raw_path}")
    print(f"Processed data path: {processed_path}")
    
    # Available balancing methods
    methods = {
        '1': 'undersample',
        '2': 'oversample', 
        '3': 'hybrid',
        '4': 'none'
    }
    
    print("\nAvailable balancing methods:")
    print("1. Undersample - Reduce majority classes to match minority")
    print("2. Oversample - Increase minority classes to match majority") 
    print("3. Hybrid - Balance to middle ground")
    print("4. None - No balancing")
    
    choice = input("\nSelect balancing method (1-4): ").strip()
    
    if choice not in methods:
        print("Invalid choice. Using undersample as default.")
        choice = '1'
    
    method = methods[choice]
    print(f"\nUsing {method} balancing method...")
    
    try:
        # Reprocess with balancing
        preprocess_sentiment(
            data_path=raw_path,
            output_path=processed_path,
            sample_fraction=1.0,  # Use full dataset
            balance_method=method
        )
        
        print("\n" + "=" * 60)
        print("BALANCING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nYou can now run the evaluation notebook to see the balanced results.")
        
    except Exception as e:
        print(f"\nError during balancing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
