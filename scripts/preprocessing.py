import os
import pandas as pd
import sys
from pathlib import Path
from typing import Optional, Tuple, Set
import logging
import re

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
RAW_DATA_DIR = config.data_dir / "raw"
PROCESSED_DATA_DIR = config.data_dir / "processed"
CACHE_DIR = config.data_dir.parent / "cache"
STOPWORDS_FILE = config.data_dir / "vietnamese-stopwords.txt"

DEFAULT_ENCODING = config.params.get("environment", {}).get("encoding", "utf-8")
RANDOM_STATE = 42


def load_stopwords(stopwords_path: str) -> Set[str]:
    """Load Vietnamese stopwords from file"""
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set(word.strip().lower() for word in f.readlines() if word.strip())
        return stopwords
    except FileNotFoundError:
        logger.warning(f"Stopwords file not found: {stopwords_path}")
        return set()
    except Exception as e:
        logger.warning(f"Error loading stopwords: {e}")
        return set()


def remove_emojis(text: str) -> str:
    """Remove emojis and emoticons from text"""
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
    """Remove stopwords from text"""
    if pd.isna(text) or not stopwords:
        return text

    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]

    return ' '.join(filtered_words)


def clean_sentiment_text(text: str, stopwords: Set[str]) -> str:
    """Clean sentiment text by removing emojis and stopwords"""
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
    """Clean summarization text (lighter cleaning to preserve meaning)"""
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
    """Load and validate CSV datasets"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path, encoding=DEFAULT_ENCODING)

    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df


def clean_and_sample_data(df: pd.DataFrame, required_columns: list,
                          sample_fraction: float, apply_text_cleaning: bool = False,
                          text_column: str = None, stopwords: Set[str] = None) -> pd.DataFrame:
    """Clean dataset by removing missing values and sample data"""
    initial_shape = df.shape

    df_clean = df.dropna(subset=required_columns)
    logger.info(f"Dropped {initial_shape[0] - df_clean.shape[0]} rows with missing values")

    if apply_text_cleaning and text_column and text_column in df_clean.columns:
        logger.info(f"Applying text cleaning to column: {text_column}")

        df_clean[text_column] = df_clean[text_column].apply(
            lambda x: clean_sentiment_text(x, stopwords)
        )

        df_clean = df_clean[df_clean[text_column].str.strip() != '']
        logger.info(f"Removed {initial_shape[0] - len(df_clean)} rows with empty text after cleaning")

    if sample_fraction < 1.0:
        df_clean = df_clean.sample(frac=sample_fraction, random_state=RANDOM_STATE).reset_index(drop=True)
        logger.info(f"Sampled {sample_fraction:.1%} of data")

    logger.info(f"Final dataset shape: {df_clean.shape}")
    return df_clean



def validate_processed_data(df: pd.DataFrame, dataset_type: str) -> bool:
    """Validate processed data quality"""
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
            
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            balance_ratio = label_counts.max() / label_counts.min()
            if balance_ratio > 2.0:
                issues.append(f"Imbalanced labels (ratio: {balance_ratio:.1f}:1)")
    
    if dataset_type == 'summarization':
        duplicate_contents = df['Text'].duplicated().sum()
        if duplicate_contents > len(df) * 0.05:
            issues.append(f"High duplicate content: {duplicate_contents}")
    elif dataset_type == 'sentiment':
        duplicate_comments = df['comment'].duplicated().sum()
        if duplicate_comments > len(df) * 0.05:
            issues.append(f"High duplicate comments: {duplicate_comments}")
    
    if issues:
        logger.warning(f"Data validation issues for {dataset_type}: {'; '.join(issues)}")
        return True
    else:
        logger.info(f"Data validation passed for {dataset_type}")
        return True


def save_processed_data(df: pd.DataFrame, output_path: str, filename: str) -> None:
    """Save processed DataFrame to CSV file"""
    Path(output_path).mkdir(parents=True, exist_ok=True)

    out_file = os.path.join(output_path, filename)
    df.to_csv(out_file, index=False, encoding=DEFAULT_ENCODING)
    logger.info(f"Saved processed data to {out_file}")


def preprocess_summarization(data_path: str, output_path: str,
                             sample_fraction: Optional[float] = None) -> None:
    """Load, sample, and clean summarization dataset"""
    logger.info("Processing summarization dataset...")

    if sample_fraction is None:
        sample_fraction = 0.5  # Default 50% sample for summarization

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

        logger.info(f"Cleaning Text and Summary columns...")
        df_sampled['Text'] = df_sampled['Text'].apply(clean_summarization_text)
        df_sampled['Summary'] = df_sampled['Summary'].apply(clean_summarization_text)

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
            logger.error("Summarization data failed validation")
            raise ValueError("Data validation failed")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error processing summarization dataset: {e}")
        raise


def preprocess_sentiment(data_path: str, output_path: str,
                         sample_fraction: Optional[float] = None) -> None:
    """Load, sample, and clean sentiment dataset with text cleaning"""
    logger.info("Processing sentiment dataset...")

    if sample_fraction is None:
        sample_fraction = 1.0  # Default 100% sample for sentiment

    file_path = os.path.join(data_path, "data_sentiment.csv")
    required_columns = ["comment", "label"]

    stopwords = load_stopwords(str(STOPWORDS_FILE))

    try:
        df = load_csv_dataset(file_path, required_columns)

        df_sampled = clean_and_sample_data(
            df,
            required_columns,
            sample_fraction,
            apply_text_cleaning=False
        )

        logger.info(f"Cleaning comment column...")
        df_sampled['comment'] = df_sampled['comment'].apply(
            lambda x: clean_sentiment_text(x, stopwords)
        )

        initial_count = len(df_sampled)
        df_processed = df_sampled[df_sampled['comment'].str.strip() != ''].reset_index(drop=True)
        df_processed = df_processed[df_processed['comment'].str.len() >= 10].reset_index(drop=True)

        removed_count = initial_count - len(df_processed)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} rows with empty/short comments after cleaning")

        logger.info(f"Final sentiment dataset shape: {df_processed.shape}")
        
        if validate_processed_data(df_processed, 'sentiment'):
            save_processed_data(df_processed, output_path, "reviews_clean.csv")
        else:
            logger.error("Sentiment data failed validation")
            raise ValueError("Data validation failed")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error processing sentiment dataset: {e}")
        raise


def initialize_model_components() -> Tuple[object, object]:
    """Initialize tokenizer and model (lazy loading)"""
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
    raw_path = str(RAW_DATA_DIR)
    processed_path = str(PROCESSED_DATA_DIR)

    try:
        logger.info("Starting data preprocessing pipeline...")
        preprocess_summarization(raw_path, processed_path)
        preprocess_sentiment(raw_path, processed_path)
        logger.info("Data preprocessing completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()