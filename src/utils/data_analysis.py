"""
Data analysis utilities for Vietnamese Text Processing
"""
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path

def load_and_sample_data(file_path: str, 
                        sample_ratio: int = 4, 
                        required_columns: List[str] = None,
                        random_seed: int = 42) -> pd.DataFrame:
    """
    Load dataset and return a sample
    
    Args:
        file_path: Path to the CSV file
        sample_ratio: Take 1/sample_ratio of the data
        required_columns: Columns that must be present and non-null
        random_seed: Random seed for reproducibility
    
    Returns:
        Sampled DataFrame
    """
    from datasets import load_dataset
    
    try:
        dataset = load_dataset('csv', data_files=file_path)
        
        train_ds = dataset["train"].shuffle(seed=random_seed)
        sample_size = len(train_ds) // sample_ratio
        train_sample = train_ds.select(range(sample_size))
        
        # Convert to DataFrame
        df = pd.DataFrame(train_sample)
        
        # Remove missing values if required columns specified
        if required_columns:
            df = df.dropna(subset=required_columns)
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error loading dataset from {file_path}: {e}")

def calculate_text_metrics(df: pd.DataFrame, 
                         text_column: str, 
                         new_column_name: str = None) -> pd.DataFrame:
    """
    Calculate various text metrics
    
    Args:
        df: Input DataFrame
        text_column: Column containing text
        new_column_name: Name for the length column
    
    Returns:
        DataFrame with added metrics
    """
    if new_column_name is None:
        new_column_name = f"{text_column}_len"
    
    # Word count
    df[new_column_name] = df[text_column].astype(str).apply(lambda x: len(x.split()))
    
    # Character count
    df[f"{text_column}_chars"] = df[text_column].astype(str).str.len()
    
    # Sentence count (approximate)
    df[f"{text_column}_sentences"] = df[text_column].astype(str).apply(
        lambda x: len(re.split(r'[.!?]+', x)) - 1
    )
    
    return df

def analyze_compression_ratio(df: pd.DataFrame, 
                            content_col: str, 
                            summary_col: str) -> pd.DataFrame:
    """
    Analyze compression ratio for summarization data
    
    Args:
        df: DataFrame with content and summary
        content_col: Column name for content length
        summary_col: Column name for summary length
    
    Returns:
        DataFrame with compression ratio
    """
    df['compression_ratio'] = df[summary_col] / df[content_col]
    return df

def get_data_quality_report(df: pd.DataFrame, 
                          text_columns: List[str]) -> Dict:
    """
    Generate data quality report
    
    Args:
        df: Input DataFrame
        text_columns: List of text columns to analyze
    
    Returns:
        Dictionary with quality metrics
    """
    report = {
        'total_samples': len(df),
        'missing_values': {},
        'duplicates': {},
        'empty_text': {}
    }
    
    for col in text_columns:
        if col in df.columns:
            report['missing_values'][col] = df[col].isnull().sum()
            report['duplicates'][col] = len(df) - df[col].nunique()
            report['empty_text'][col] = (df[col].astype(str).str.strip() == '').sum()
    
    return report

def find_extreme_cases(df: pd.DataFrame, 
                      length_column: str, 
                      text_column: str,
                      n_cases: int = 3) -> Dict:
    """
    Find extreme cases (longest/shortest texts)
    
    Args:
        df: Input DataFrame
        length_column: Column with text lengths
        text_column: Column with text content
        n_cases: Number of cases to return
    
    Returns:
        Dictionary with extreme cases
    """
    return {
        'longest': df.nlargest(n_cases, length_column)[text_column].tolist(),
        'shortest': df.nsmallest(n_cases, length_column)[text_column].tolist(),
        'longest_lengths': df.nlargest(n_cases, length_column)[length_column].tolist(),
        'shortest_lengths': df.nsmallest(n_cases, length_column)[length_column].tolist()
    }

def validate_vietnamese_text(text: str) -> bool:
    """
    Validate if text contains valid Vietnamese characters
    
    Args:
        text: Input text
    
    Returns:
        Boolean indicating if text is valid Vietnamese
    """
    vietnamese_pattern = r'^[a-zA-Z0-9\s\.,!?\(\)\[\]àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]+$'
    return bool(re.match(vietnamese_pattern, str(text)))

def analyze_label_distribution(df: pd.DataFrame, 
                             label_column: str) -> Dict:
    """
    Analyze label distribution for classification data
    
    Args:
        df: Input DataFrame
        label_column: Column with labels
    
    Returns:
        Dictionary with distribution metrics
    """
    label_counts = df[label_column].value_counts()
    label_percentages = df[label_column].value_counts(normalize=True) * 100
    
    return {
        'counts': label_counts.to_dict(),
        'percentages': label_percentages.round(2).to_dict(),
        'imbalance_ratio': label_counts.max() / label_counts.min(),
        'unique_labels': df[label_column].nunique()
    }

def print_data_summary(df: pd.DataFrame, 
                      dataset_name: str,
                      text_columns: List[str]) -> None:
    """
    Print comprehensive data summary
    
    Args:
        df: Input DataFrame
        dataset_name: Name of the dataset
        text_columns: List of text columns to analyze
    """
    print(f"\n{dataset_name} Dataset Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Quality report
    quality_report = get_data_quality_report(df, text_columns)
    print(f"\nData Quality:")
    for col in text_columns:
        if col in quality_report['missing_values']:
            print(f"  {col}: {quality_report['missing_values'][col]} missing, "
                  f"{quality_report['duplicates'][col]} duplicates, "
                  f"{quality_report['empty_text'][col]} empty")
    
    # Length statistics
    for col in text_columns:
        len_col = f"{col}_len"
        if len_col in df.columns:
            print(f"\n{col.title()} Length Statistics:")
            print(df[len_col].describe())
