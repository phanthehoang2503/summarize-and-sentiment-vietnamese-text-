"""
Simple data analysis utilities for Vietnamese Text Processing demo
Focused on essential functionality with robust error handling
"""
import pandas as pd
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def load_demo_data(file_path: str, sample_size: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Simple data loading for demo purposes
    
    Args:
        file_path: Path to the CSV file
        sample_size: Maximum number of rows to load (None for all)
    
    Returns:
        DataFrame or None if loading fails
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        # Simple CSV loading with error handling
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            # Try different encodings commonly used for Vietnamese
            try:
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='cp1252')
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Sample data if requested
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled down to {len(df)} rows")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return None

def get_text_stats(text: str) -> Dict[str, int]:
    """Get basic statistics for a text string"""
    if not isinstance(text, str):
        return {'chars': 0, 'words': 0, 'sentences': 0}
    
    try:
        chars = len(text)
        words = len(text.split()) if text.strip() else 0
        # Simple sentence count (approximate)
        sentences = len([s for s in text.split('.') if s.strip()]) if text.strip() else 0
        
        return {
            'chars': chars,
            'words': words, 
            'sentences': sentences
        }
        
    except Exception as e:
        logger.error(f"Error calculating text stats: {e}")
        return {'chars': 0, 'words': 0, 'sentences': 0}

def analyze_dataset_text(df: pd.DataFrame, text_column: str) -> Dict:
    """
    Simple analysis of text data in a DataFrame
    
    Args:
        df: DataFrame containing text data
        text_column: Name of the column containing text
        
    Returns:
        Dictionary with basic statistics
    """
    if df is None or df.empty:
        return {'error': 'Empty or invalid dataset'}
    
    if text_column not in df.columns:
        return {'error': f'Column {text_column} not found'}
    
    try:
        # Remove null values
        text_data = df[text_column].dropna()
        
        if text_data.empty:
            return {'error': 'No valid text data found'}
        
        # Calculate basic stats
        total_texts = len(text_data)
        
        # Calculate text lengths
        lengths = [len(str(text)) for text in text_data]
        
        stats = {
            'total_texts': total_texts,
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'empty_texts': total_texts - len([l for l in lengths if l > 0])
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error analyzing dataset: {e}")
        return {'error': f'Analysis failed: {str(e)}'}

def create_sample_dataset(texts: List[str], labels: Optional[List] = None) -> pd.DataFrame:
    """Create a simple DataFrame from text and optional labels"""
    try:
        data = {'text': texts}
        
        if labels:
            if len(labels) != len(texts):
                logger.warning("Labels and texts have different lengths, ignoring labels")
            else:
                data['label'] = labels
        
        return pd.DataFrame(data)
        
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        return pd.DataFrame()  # Return empty DataFrame