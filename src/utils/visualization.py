"""
Visualization utilities for Vietnamese Text Analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.family"] = "DejaVu Sans"

def plot_length_distributions(df: pd.DataFrame, 
                            content_col: str, 
                            summary_col: str = None,
                            title: str = "Length Distributions") -> None:
    """Plot length distributions for text data"""
    
    if summary_col:
        # Plot both content and summary lengths
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
        
        # Contents histogram
        ax[0].hist(df[f"{content_col}_len"], bins=50, color='steelblue', alpha=0.7)
        ax[0].set_title(f"{content_col.title()} Length Distribution")
        ax[0].set_ylabel("Frequency")
        
        # Summary histogram
        ax[1].hist(df[f"{summary_col}_len"], bins=50, color='chocolate', alpha=0.7)
        ax[1].set_title(f"{summary_col.title()} Length Distribution")
        ax[1].set_xlabel("Words")
        ax[1].set_ylabel("Frequency")
    else:
        # Plot single distribution
        plt.figure(figsize=(10, 5))
        plt.hist(df[f"{content_col}_len"], bins=50, color='steelblue', alpha=0.7)
        plt.title(f"{content_col.title()} Length Distribution")
        plt.xlabel("Words")
        plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

def plot_boxplots(df: pd.DataFrame, 
                 columns: List[str], 
                 title: str = "Length Distributions") -> None:
    """Plot boxplots for length distributions"""
    
    fig, ax = plt.subplots(len(columns), 1, figsize=(10, 3 * len(columns)))
    
    if len(columns) == 1:
        ax = [ax]
    
    for i, col in enumerate(columns):
        df[[col]].plot(kind="box", vert=False, ax=ax[i])
        ax[i].set_title(f"{col.replace('_', ' ').title()} Distribution")
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_compression_ratio(df: pd.DataFrame) -> None:
    """Plot compression ratio distribution for summarization data"""
    
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='compression_ratio', bins=50, alpha=0.7)
    plt.title("Distribution of Summary Compression Ratios")
    plt.xlabel("Compression Ratio (Summary Length / Content Length)")
    plt.ylabel("Count")
    plt.axvline(df['compression_ratio'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["compression_ratio"].mean():.3f}')
    plt.legend()
    plt.show()

def plot_sentiment_distribution(df: pd.DataFrame, 
                               label_col: str = "label",
                               text_len_col: str = "text_len") -> None:
    """Plot sentiment analysis visualizations"""
    
    plt.figure(figsize=(12, 5))
    
    # Label distribution
    plt.subplot(1, 2, 1)
    df[label_col].value_counts().plot(kind="bar", color='skyblue')
    plt.title("Sentiment Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    
    # Text length by sentiment
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x=label_col, y=text_len_col, palette="Set2")
    plt.title("Text Length by Sentiment")
    plt.xlabel("Sentiment Label")
    plt.ylabel("Word Count")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_cross_dataset_comparison(df_summ: pd.DataFrame, 
                                df_sent: pd.DataFrame) -> None:
    """Compare text lengths across datasets"""
    
    plt.figure(figsize=(15, 5))
    
    # Summarization content length
    plt.subplot(1, 3, 1)
    sns.histplot(data=df_summ, x='contents_len', bins=50, alpha=0.7, color='blue')
    plt.title("Summarization: Content Length")
    plt.xlabel("Words")
    
    # Summarization summary length
    plt.subplot(1, 3, 2)
    sns.histplot(data=df_summ, x='summary_len', bins=50, alpha=0.7, color='green')
    plt.title("Summarization: Summary Length")
    plt.xlabel("Words")
    
    # Sentiment comment length
    plt.subplot(1, 3, 3)
    sns.histplot(data=df_sent, x='text_len', bins=50, alpha=0.7, color='orange')
    plt.title("Sentiment: Comment Length")
    plt.xlabel("Words")
    
    plt.tight_layout()
    plt.show()

def save_plot(filename: str, output_dir: str = "plots") -> None:
    """Save current plot to file"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
