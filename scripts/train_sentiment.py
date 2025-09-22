#!/usr/bin/env python3
"""
Vietnamese Sentiment Analysis Training Script
Train PhoBERT-based sentiment classifier on Vietnamese text data
"""
import os
import sys
import logging
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset
import numpy as np

from app.core.config import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "training.log")
    ]
)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    model_name: str = "vinai/phobert-base"
    num_labels: int = 3
    max_length: int = 256
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    early_stopping_patience: int = 3
    test_size: float = 0.2
    random_state: int = 42
    
    # Paths
    data_file: Path = PROJECT_ROOT / "data" / "processed" / "reviews_clean.csv"
    output_dir: Path = PROJECT_ROOT / "models" / "sentiment"
    cache_dir: Path = PROJECT_ROOT / "cache"
    logs_dir: Path = PROJECT_ROOT / "logs"


class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_and_prepare_data(config: TrainingConfig) -> tuple:
    """Load and prepare sentiment data for training"""
    logger.info(f"Loading data from {config.data_file}")
    
    if not config.data_file.exists():
        raise FileNotFoundError(f"Data file not found: {config.data_file}")
    
    # Load data
    df = pd.read_csv(config.data_file)
    logger.info(f"Loaded {len(df)} samples")
    
    # Check required columns
    required_columns = ['comment', 'label']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Clean data
    df = df.dropna(subset=required_columns)
    df = df[df['comment'].str.strip() != '']
    df = df[df['comment'].str.len() >= 5]  # Minimum length
    
    logger.info(f"After cleaning: {len(df)} samples")
    
    # Label mapping and validation
    label_mapping = {'NEG': 0, 'NEU': 1, 'POS': 2}
    if df['label'].dtype == 'object':
        df['label'] = df['label'].map(label_mapping)
    
    # Remove invalid labels
    df = df[df['label'].isin([0, 1, 2])]
    
    # Check label distribution
    label_counts = df['label'].value_counts().sort_index()
    logger.info(f"Label distribution: {dict(label_counts)}")
    
    # Prepare texts and labels
    texts = df['comment'].tolist()
    labels = df['label'].tolist()
    
    # Train-test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, 
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels
    )
    
    logger.info(f"Train samples: {len(train_texts)}, Test samples: {len(test_texts)}")
    
    return train_texts, test_texts, train_labels, test_labels


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }


def train_sentiment_model(config: TrainingConfig):
    """Main training function"""
    logger.info("Starting sentiment model training...")
    logger.info(f"Configuration: {config}")
    
    # Create output directories
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_texts, test_texts, train_labels, test_labels = load_and_prepare_data(config)
    
    # Initialize tokenizer and model
    logger.info(f"Loading tokenizer and model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, 
        cache_dir=str(config.cache_dir)
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        cache_dir=str(config.cache_dir)
    )
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, config.max_length)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, config.max_length)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_dir=str(config.logs_dir),
        logging_steps=100,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        report_to=None,  # Disable wandb/tensorboard
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate the model
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # Save the final model
    logger.info("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Generate detailed classification report
    logger.info("Generating classification report...")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    label_names = ['NEG', 'NEU', 'POS']
    report = classification_report(test_labels, y_pred, target_names=label_names, digits=4)
    logger.info(f"Classification Report:\n{report}")
    
    # Save report to file
    report_file = config.output_dir / "classification_report.txt"
    with open(report_file, 'w') as f:
        f.write(f"Training completed at: {datetime.now()}\n")
        f.write(f"Model: {config.model_name}\n")
        f.write(f"Training samples: {len(train_texts)}\n")
        f.write(f"Test samples: {len(test_texts)}\n")
        f.write(f"Final evaluation results: {eval_results}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    logger.info(f"Training completed! Model saved to: {config.output_dir}")
    logger.info(f"Classification report saved to: {report_file}")
    
    return trainer, eval_results


def main():
    """Main function"""
    try:
        # Initialize configuration
        config = TrainingConfig()
        
        # Check if data file exists
        if not config.data_file.exists():
            logger.error(f"Data file not found: {config.data_file}")
            logger.error("Please run preprocessing first: python scripts/preprocessing.py")
            return
        
        # Check GPU availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Start training
        trainer, results = train_sentiment_model(config)
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()