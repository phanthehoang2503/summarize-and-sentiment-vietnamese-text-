import os
import pandas as pd
import sys
from pathlib import Path
from typing import Optional, Tuple, Set
import logging
import re
from sklearn.utils import resample

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Add project root to path for legacy compatibility
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_config

logger = logging.getLogger(__name__)

# Get configuration
config = get_config()
DATA_DIR = config.data_dir
RAW_DATA_DIR = config.raw_data_dir
PROCESSED_DATA_DIR = config.processed_data_dir
CACHE_DIR = config.cache_dir
STOPWORDS_FILE = config.stopwords_file

DEFAULT_ENCODING = config.get_preprocessing_config().get("general", {}).get("encoding", "utf-8-sig")
RANDOM_STATE = config.get_preprocessing_config().get("general", {}).get("random_state", 42)


def load_stopwords(stopwords_path: str) -> Set[str]:
    """
    Load Vietnamese stopwords from file

    Args:
        stopwords_path: Path to the stopwords file

    Returns:
        Set of stopwords
    """
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set(word.strip().lower() for word in f.readlines() if word.strip())
        logger.info(f"Loaded {len(stopwords)} stopwords from {stopwords_path}")
        return stopwords
    except FileNotFoundError:
        logger.warning(f"Stopwords file not found: {stopwords_path}. Proceeding without stopword removal.")
        return set()
    except Exception as e:
        logger.warning(f"Error loading stopwords: {e}. Proceeding without stopword removal.")
        return set()


def remove_emojis(text: str) -> str:
    """
    Remove emojis and emoticons from text

    Args:
        text: Input text

    Returns:
        Text with emojis removed
    """
    if pd.isna(text):
        return text

    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        flags=re.UNICODE
    )

    text = emoji_pattern.sub(r'', text)

    emoticon_pattern = re.compile(
        r'[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]|[\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*\']?[:;=8]')
    text = emoticon_pattern.sub(r'', text)

    return text.strip()


def remove_stopwords(text: str, stopwords: Set[str]) -> str:
    """
    Remove stopwords from text

    Args:
        text: Input text
        stopwords: Set of stopwords to remove

    Returns:
        Text with stopwords removed
    """
    if pd.isna(text) or not stopwords:
        return text

    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]

    return ' '.join(filtered_words)


def clean_sentiment_text(text: str, stopwords: Set[str]) -> str:
    """
    Clean sentiment text by removing emojis and stopwords

    Args:
        text: Input text
        stopwords: Set of stopwords to remove

    Returns:
        Cleaned text
    """
    pattern = r'(.)\1{2,}'

    if pd.isna(text):
        return text

    text = remove_emojis(text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[""''"\']+', '', text)
    text = remove_stopwords(text, stopwords)
    text = re.sub(pattern, r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_summarization_text(text: str) -> str:
    """
    Clean summarization text (lighter cleaning to preserve meaning)

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    if pd.isna(text):
        return text

    text = re.sub(r'\s+', ' ', text).strip()
    
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_csv_dataset(file_path: str, required_columns: list) -> pd.DataFrame:
    """
    Generic function to load and validate CSV datasets

    Args:
        file_path: Path to the CSV file
        required_columns: List of required column names

    Returns:
        pd.DataFrame: Loaded and validated DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path, encoding=DEFAULT_ENCODING)

    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df


def clean_and_sample_data(df: pd.DataFrame, required_columns: list,
                          sample_fraction: float, apply_text_cleaning: bool = False,
                          text_column: str = None, stopwords: Set[str] = None) -> pd.DataFrame:
    """
    Clean dataset by removing missing values and sample data

    Args:
        df: Input DataFrame
        required_columns: Columns that must not have missing values
        sample_fraction: Fraction of data to sample (between 0 and 1)
        apply_text_cleaning: Whether to apply text cleaning (emoji/stopword removal)
        text_column: Column name to apply text cleaning to
        stopwords: Set of stopwords for removal

    Returns:
        pd.DataFrame: Cleaned and sampled DataFrame
    """
    initial_shape = df.shape

    # Drop rows with missing values in required columns
    df_clean = df.dropna(subset=required_columns)
    logger.info(f"Dropped {initial_shape[0] - df_clean.shape[0]} rows with missing values")

    if apply_text_cleaning and text_column and text_column in df_clean.columns:
        logger.info(f"Applying text cleaning to column: {text_column}")

        if not df_clean.empty:
            # Use safe logging for Vietnamese text
            sample_before = df_clean[text_column].iloc[0][:100]
            logger.info(f"Sample text before cleaning: {len(sample_before)} characters")

        df_clean[text_column] = df_clean[text_column].apply(
            lambda x: clean_sentiment_text(x, stopwords)
        )

        if not df_clean.empty:
            # Use safe logging for Vietnamese text
            sample_after = df_clean[text_column].iloc[0][:100]
            logger.info(f"Sample text after cleaning: {len(sample_after)} characters")

        df_clean = df_clean[df_clean[text_column].str.strip() != '']
        logger.info(f"Removed {initial_shape[0] - len(df_clean)} rows with empty text after cleaning")

    if sample_fraction < 1.0:
        df_clean = df_clean.sample(frac=sample_fraction, random_state=RANDOM_STATE).reset_index(drop=True)
        logger.info(f"Sampled {sample_fraction:.1%} of data")

    logger.info(f"Final dataset shape: {df_clean.shape}")
    return df_clean


def balance_sentiment_dataset(df: pd.DataFrame, method: str = 'undersample', target_size: Optional[int] = None) -> pd.DataFrame:
    """
    Balance sentiment dataset using different strategies
    
    Args:
        df: Input DataFrame with 'label' column
        method: Balancing method ('undersample', 'oversample', 'hybrid')
        target_size: Target size per class (if None, uses smallest class for undersample)
        
    Returns:
        Balanced DataFrame
    """
    logger.info(f"Balancing dataset using {method} method...")
    
    # Get class distribution
    class_counts = df['label'].value_counts()
    logger.info(f"Original class distribution:\n{class_counts}")
    
    min_class_size = class_counts.min()
    max_class_size = class_counts.max()
    
    if method == 'undersample':
        # Downsample majority classes to match minority class
        target_size = target_size or min_class_size
        
        balanced_dfs = []
        for label in class_counts.index:
            class_df = df[df['label'] == label]
            if len(class_df) > target_size:
                # Downsample
                downsampled = resample(class_df, 
                                     replace=False, 
                                     n_samples=target_size, 
                                     random_state=RANDOM_STATE)
                balanced_dfs.append(downsampled)
                logger.info(f"Downsampled {label}: {len(class_df)} -> {target_size}")
            else:
                balanced_dfs.append(class_df)
                logger.info(f"Kept {label}: {len(class_df)} (already smaller)")
                
    elif method == 'oversample':
        # Upsample minority classes to match majority class
        target_size = target_size or max_class_size
        
        balanced_dfs = []
        for label in class_counts.index:
            class_df = df[df['label'] == label]
            if len(class_df) < target_size:
                # Oversample
                upsampled = resample(class_df, 
                                   replace=True, 
                                   n_samples=target_size, 
                                   random_state=RANDOM_STATE)
                balanced_dfs.append(upsampled)
                logger.info(f"Upsampled {label}: {len(class_df)} -> {target_size}")
            else:
                balanced_dfs.append(class_df)
                logger.info(f"Kept {label}: {len(class_df)} (already larger)")
                
    elif method == 'hybrid':
        # Balance to a middle ground (average of min and max)
        target_size = target_size or int((min_class_size + max_class_size) / 2)
        
        balanced_dfs = []
        for label in class_counts.index:
            class_df = df[df['label'] == label]
            if len(class_df) > target_size:
                # Downsample
                downsampled = resample(class_df, 
                                     replace=False, 
                                     n_samples=target_size, 
                                     random_state=RANDOM_STATE)
                balanced_dfs.append(downsampled)
                logger.info(f"Downsampled {label}: {len(class_df)} -> {target_size}")
            elif len(class_df) < target_size:
                # Upsample
                upsampled = resample(class_df, 
                                   replace=True, 
                                   n_samples=target_size, 
                                   random_state=RANDOM_STATE)
                balanced_dfs.append(upsampled)
                logger.info(f"Upsampled {label}: {len(class_df)} -> {target_size}")
            else:
                balanced_dfs.append(class_df)
                logger.info(f"Kept {label}: {len(class_df)} (already balanced)")
    
    else:
        raise ValueError(f"Unknown balancing method: {method}")
    
    # Combine all balanced classes
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    # Log final distribution
    final_counts = balanced_df['label'].value_counts()
    logger.info(f"Balanced class distribution:\n{final_counts}")
    
    balance_ratio = final_counts.max() / final_counts.min()
    logger.info(f"Final balance ratio: {balance_ratio:.1f}:1")
    
    return balanced_df


def validate_processed_data(df: pd.DataFrame, dataset_type: str) -> bool:
    """
    Validate processed data quality
    
    Args:
        df: Processed DataFrame
        dataset_type: Type of dataset ('summarization' or 'sentiment')
        
    Returns:
        bool: True if data passes validation
    """
    issues = []
    
    if df.empty:
        issues.append("Dataset is empty")
        
    if dataset_type == 'summarization':
        short_contents = (df['Text'].str.len() < 50).sum()
        short_summaries = (df['Summary'].str.len() < 10).sum()
        
        if short_contents > len(df) * 0.3:
            issues.append(f"Too many short contents: {short_contents}")
        if short_summaries > len(df) * 0.1:
            issues.append(f"Too many short summaries: {short_summaries}")
            
    elif dataset_type == 'sentiment':
        short_comments = (df['comment'].str.len() < 10).sum()
        if short_comments > len(df) * 0.1:
            issues.append(f"Too many short comments: {short_comments}")
            
        # Check label balance
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            balance_ratio = label_counts.max() / label_counts.min()
            if balance_ratio > 2.0:  # More lenient threshold
                issues.append(f"Imbalanced labels (ratio: {balance_ratio:.1f}:1)")
    
    # Check for duplicates
    if dataset_type == 'summarization':
        duplicate_contents = df['Text'].duplicated().sum()
        if duplicate_contents > len(df) * 0.05:  # More than 5% duplicates
            issues.append(f"High duplicate content: {duplicate_contents}")
    elif dataset_type == 'sentiment':
        duplicate_comments = df['comment'].duplicated().sum()
        if duplicate_comments > len(df) * 0.05:
            issues.append(f"High duplicate comments: {duplicate_comments}")
    
    if issues:
        logger.warning(f"Data validation issues for {dataset_type}: {'; '.join(issues)}")
        return True  # Still return True to proceed, but with warnings
    else:
        logger.info(f"Data validation passed for {dataset_type}")
        return True


def save_processed_data(df: pd.DataFrame, output_path: str, filename: str) -> None:
    """
    Save processed DataFrame to CSV file

    Args:
        df: DataFrame to save
        output_path: Output directory path
        filename: Output filename
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)

    out_file = os.path.join(output_path, filename)
    df.to_csv(out_file, index=False, encoding=DEFAULT_ENCODING)
    logger.info(f"Saved processed data to {out_file}")


def preprocess_summarization(data_path: str, output_path: str,
                             sample_fraction: Optional[float] = None) -> None:
    """
    Load, sample, and clean summarization dataset with text cleaning

    Args:
        data_path: Path to raw data directory  
        output_path: Path to processed data directory
        sample_fraction: Fraction of data to sample (from config if None)
    """
    logger.info("Processing summarization dataset...")

    if sample_fraction is None:
        preprocessing_config = config.get_preprocessing_config("summarization")
        sample_fraction = preprocessing_config.get("sample_fraction", 0.5)  # Default 100%

    file_path = os.path.join(data_path, "data_summary.csv")
    required_columns = ["Text", "Summary"]

    try:
        df = load_csv_dataset(file_path, required_columns)

        df = df[required_columns]

        df_sampled = clean_and_sample_data(
            df,
            required_columns,
            sample_fraction,
            apply_text_cleaning=False
        )

        logger.info(f"Now cleaning the sampled dataset of {len(df_sampled)} rows...")

        #cleaning content (use lighter cleaning for summarization)
        logger.info("Cleaning Text column...")
        if not df_sampled.empty:
            # Use safe logging for Vietnamese text
            sample_text = df_sampled['Text'].iloc[0][:100]
            logger.info(f"Sample Text before cleaning: {len(sample_text)} characters")

            df_sampled['Text'] = df_sampled['Text'].apply(clean_summarization_text)

            sample_text_after = df_sampled['Text'].iloc[0][:100] 
            logger.info(f"Sample Text after cleaning: {len(sample_text_after)} characters")

        logger.info("Cleaning Summary column...")
        if not df_sampled.empty:
            # Use safe logging for Vietnamese text
            sample_summary = df_sampled['Summary'].iloc[0][:100]
            logger.info(f"Sample Summary before cleaning: {len(sample_summary)} characters")

            df_sampled['Summary'] = df_sampled['Summary'].apply(clean_summarization_text)

            sample_summary_after = df_sampled['Summary'].iloc[0][:100]
            logger.info(f"Sample Summary after cleaning: {len(sample_summary_after)} characters")

        initial_count = len(df_sampled)
        df_processed = df_sampled[
            (df_sampled['Text'].str.strip() != '') &
            (df_sampled['Summary'].str.strip() != '')
            ].reset_index(drop=True)

        removed_count = initial_count - len(df_processed)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} rows with empty text after cleaning")

        logger.info(f"Final summarization dataset shape: {df_processed.shape}")
        
        if validate_processed_data(df_processed, 'summarization'):
            save_processed_data(df_processed, output_path, "summary_clean.csv")
        else:
            logger.error("Summarization data failed validation - check processing steps")
            raise ValueError("Data validation failed")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error processing summarization dataset: {e}")
        raise


def preprocess_sentiment(data_path: str, output_path: str,
                         sample_fraction: Optional[float] = None,
                         balance_method: Optional[str] = None) -> None:
    """
    Load, sample, and clean sentiment dataset with text cleaning and optional balancing

    Args:
        data_path: Path to raw data directory
        output_path: Path to processed data directory
        sample_fraction: Fraction of data to sample (from config if None)
        balance_method: Balancing method ('undersample', 'oversample', 'hybrid', None)
    """
    logger.info("Processing sentiment dataset...")

    if sample_fraction is None:
        preprocessing_config = config.get_preprocessing_config("sentiment")
        sample_fraction = preprocessing_config.get("sample_fraction", 1.0)  # Default 100%

    if balance_method is None:
        preprocessing_config = config.get_preprocessing_config("sentiment")
        balance_method = preprocessing_config.get("balance_method", "undersample")  # Default undersample

    file_path = os.path.join(data_path, "data_sentiment.csv")
    required_columns = ["comment", "label"]

    stopwords = load_stopwords(str(STOPWORDS_FILE))

    try:
        df = load_csv_dataset(file_path, required_columns)

        df_sampled = clean_and_sample_data(
            df,
            required_columns,
            sample_fraction,
            apply_text_cleaning=False  # No cleaning yet, just sampling
        )

        logger.info(f"Now cleaning the sampled dataset of {len(df_sampled)} rows...")

        logger.info("Cleaning comment column...")
        if not df_sampled.empty:
            # Use safe logging for Vietnamese text
            sample_comment = df_sampled['comment'].iloc[0][:100] 
            logger.info(f"Sample comment before cleaning: {len(sample_comment)} characters")

            df_sampled['comment'] = df_sampled['comment'].apply(
                lambda x: clean_sentiment_text(x, stopwords)
            )

            sample_comment_after = df_sampled['comment'].iloc[0][:100]
            logger.info(f"Sample comment after cleaning: {len(sample_comment_after)} characters")

        initial_count = len(df_sampled)
        df_processed = df_sampled[df_sampled['comment'].str.strip() != ''].reset_index(drop=True)

        # additionally remove very short comments (less than 10 chars)
        df_processed = df_processed[df_processed['comment'].str.len() >= 10].reset_index(drop=True)

        removed_count = initial_count - len(df_processed)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} rows with empty/short comments after cleaning")

        # Apply balancing if specified
        if balance_method and balance_method != 'none':
            logger.info(f"Applying {balance_method} balancing...")
            df_processed = balance_sentiment_dataset(df_processed, method=balance_method)

        logger.info(f"Final sentiment dataset shape: {df_processed.shape}")
        
        if validate_processed_data(df_processed, 'sentiment'):
            save_processed_data(df_processed, output_path, "reviews_clean.csv")
        else:
            logger.error("Sentiment data failed validation - check processing steps")
            raise ValueError("Data validation failed")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error processing sentiment dataset: {e}")
        raise


def initialize_model_components() -> Tuple[object, object]:
    """
    Initialize tokenizer and model (lazy loading)

    Returns:
        Tuple of (tokenizer, model)
    """
    logger.info("Initializing model components...")

    try:
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        tokenizer = AutoTokenizer.from_pretrained(
            "vinai/phobert-base",
            cache_dir=str(CACHE_DIR)
        )
        model = AutoModelForMaskedLM.from_pretrained(
            "vinai/phobert-base",
            cache_dir=str(CACHE_DIR)
        )

        logger.info("Model components initialized successfully")
        return tokenizer, model

    except Exception as e:
        logger.error(f"Error initializing model components: {e}")
        raise


def main():
    """Main function to run data preprocessing pipeline"""
    # Use absolute paths
    raw_path = str(RAW_DATA_DIR)
    processed_path = str(PROCESSED_DATA_DIR)

    try:
        logger.info("Starting data preprocessing pipeline...")
        logger.info(f"Raw data path: {raw_path}")
        logger.info(f"Processed data path: {processed_path}")

        # Process datasets
        preprocess_summarization(raw_path, processed_path)
        preprocess_sentiment(raw_path, processed_path)

        # tokenizer, model = initialize_model_components()

        logger.info("Data preprocessing completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()