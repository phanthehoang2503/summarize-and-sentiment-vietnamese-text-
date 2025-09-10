# src/models/sentiment.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config


class VietnameseSentimentAnalyzer:
    """
    Vietnamese Sentiment Analysis Model using PhoBERT
    
    This class provides sentiment analysis for Vietnamese text using a fine-tuned PhoBERT model.
    Supports binary sentiment classification (positive/negative) or multi-class sentiment.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize Vietnamese Sentiment Analyzer
        
        Args:
            model_path: Path to the trained sentiment model (defaults to config)
            cache_dir: Cache directory for models (defaults to config)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        if model_path is None:
            model_path = config.sentiment_model_dir
        if cache_dir is None:
            cache_dir = config.sentiment_cache_dir
            
        self.model_path = Path(model_path)
        self.cache_dir = Path(cache_dir)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        print(f"Model path: {self.model_path}")
        
        self.model = None
        self.tokenizer = None
        self.label_mapping = {
            0: "NEG",
            1: "NEU",
            2: "POS"
        }
        
    def load_model(self):
        """Load the sentiment model and tokenizer"""
        if self.model is None:
            print("Loading sentiment model...")
            
            try:
                # Import here to avoid dependency issues at module level
                import os
                os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN for compatibility
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    cache_dir=self.cache_dir,
                    local_files_only=True  # Use only local files
                )
                
                # Load model
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path,
                    cache_dir=self.cache_dir,
                    local_files_only=True  # Use only local files
                )
                
                # Move model to device
                self.model.to(self.device)
                self.model.eval()
                
                print(f"✅ Sentiment model loaded successfully")
                print(f"   Model: {self.model_path}")
                print(f"   Vocab size: {len(self.tokenizer)}")
                
            except Exception as e:
                print(f"❌ Error loading sentiment model: {e}")
                print(f"   Make sure the model files exist at: {self.model_path}")
                raise
                
    def predict_sentiment(self, 
                         text: str, 
                         return_probabilities: bool = False) -> Union[str, Dict]:
        """
        Predict sentiment for a single text
        
        Args:
            text: Input Vietnamese text
            return_probabilities: If True, return probabilities for all classes
            
        Returns:
            Predicted sentiment label or dict with probabilities
        """
        if self.model is None:
            self.load_model()
            
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            
        if return_probabilities:
            return {
                'predicted_label': self.label_mapping[predicted_class],
                'predicted_class': predicted_class,
                'probabilities': {
                    label: prob.item() 
                    for label, prob in zip(self.label_mapping.values(), probabilities[0])
                },
                'confidence': probabilities[0][predicted_class].item()
            }
        else:
            return self.label_mapping[predicted_class]
    
    def predict_batch(self, 
                     texts: List[str], 
                     batch_size: int = 16,
                     return_probabilities: bool = False) -> List[Union[str, Dict]]:
        """
        Predict sentiments for a batch of texts
        
        Args:
            texts: List of Vietnamese texts
            batch_size: Batch size for processing
            return_probabilities: If True, return probabilities for all classes
            
        Returns:
            List of predicted sentiment labels or dicts with probabilities
        """
        if self.model is None:
            self.load_model()
            
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_classes = torch.argmax(probabilities, dim=-1)
                
            # Process results
            for j, pred_class in enumerate(predicted_classes):
                pred_class = pred_class.item()
                
                if return_probabilities:
                    results.append({
                        'predicted_label': self.label_mapping[pred_class],
                        'predicted_class': pred_class,
                        'probabilities': {
                            label: prob.item() 
                            for label, prob in zip(self.label_mapping.values(), probabilities[j])
                        },
                        'confidence': probabilities[j][pred_class].item()
                    })
                else:
                    results.append(self.label_mapping[pred_class])
                    
        return results
    
    def analyze_dataset(self, 
                       dataset_path: Optional[str] = None,
                       text_column: str = 'text',
                       batch_size: int = 16,
                       sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Analyze sentiment for a dataset
        
        Args:
            dataset_path: Path to dataset CSV file (defaults to config)
            text_column: Name of the text column
            batch_size: Batch size for processing
            sample_size: Number of samples to analyze (None for all)
            
        Returns:
            DataFrame with sentiment predictions
        """
        # Load dataset
        if dataset_path is None:
            dataset_path = config.sentiment_data
            
        print(f"Loading dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} samples")
        
        # Sample data if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampled {len(df)} samples for analysis")
            
        # Get predictions
        texts = df[text_column].tolist()
        predictions = self.predict_batch(texts, batch_size=batch_size, return_probabilities=True)
        
        # Add predictions to dataframe
        df = df.copy()
        df['predicted_sentiment'] = [pred['predicted_label'] for pred in predictions]
        df['confidence'] = [pred['confidence'] for pred in predictions]
        df['pos_prob'] = [pred['probabilities']['POS'] for pred in predictions]
        df['neg_prob'] = [pred['probabilities']['NEG'] for pred in predictions]
        df['neu_prob'] = [pred['probabilities']['NEU'] for pred in predictions]
        
        return df
    
    def evaluate_model(self, 
                      dataset_path: Optional[str] = None,
                      text_column: str = 'text',
                      label_column: str = 'label',
                      batch_size: int = 16) -> Dict:
        """
        Evaluate model performance on a labeled dataset
        
        Args:
            dataset_path: Path to labeled dataset
            text_column: Name of the text column
            label_column: Name of the label column
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Load dataset
        if dataset_path is None:
            dataset_path = config.sentiment_data
            
        df = pd.read_csv(dataset_path)
        
        # Get predictions
        predictions = self.predict_batch(df[text_column].tolist(), batch_size=batch_size)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        true_labels = df[label_column].tolist()
        
        # Convert labels to strings if needed
        if isinstance(true_labels[0], (int, float)):
            true_labels = [self.label_mapping[int(label)] for label in true_labels]
            
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(true_labels, predictions).tolist()
        }


def create_sentiment_analyzer(checkpoint: str = "latest") -> VietnameseSentimentAnalyzer:
    """
    Create a Vietnamese sentiment analyzer with specified checkpoint
    
    Args:
        checkpoint: Checkpoint to use ("latest", "1182", "788")
        
    Returns:
        VietnameseSentimentAnalyzer instance
    """
    if checkpoint == "1182":
        model_path = config.sentiment_model_dir / "checkpoint-1182"
    elif checkpoint == "788":
        model_path = config.sentiment_model_dir / "checkpoint-788"
    else:
        model_path = config.sentiment_model_dir
        
    return VietnameseSentimentAnalyzer(model_path=model_path)


def quick_sentiment_analysis(text: str, checkpoint: str = "latest") -> str:
    """
    Quick sentiment analysis for a single text
    
    Args:
        text: Vietnamese text to analyze
        checkpoint: Model checkpoint to use
        
    Returns:
        Predicted sentiment label
    """
    analyzer = create_sentiment_analyzer(checkpoint)
    return analyzer.predict_sentiment(text)


if __name__ == "__main__":  
    # Create analyzer
    analyzer = create_sentiment_analyzer()
    
    # Test with sample texts
    test_texts = [
        "Tôi rất hài lòng với sản phẩm này, chất lượng tuyệt vời!",
        "Dịch vụ khách hàng tệ quá, tôi rất thất vọng.",
        "Món ăn ngon, giá cả hợp lý, nhân viên thân thiện.",
        "Chất lượng sản phẩm không như mong đợi, rất tệ."
    ]
    
    print("\nsample sentiment analysis:")
    for text in test_texts:
        result = analyzer.predict_sentiment(text, return_probabilities=True)
        print(f"Text: {text}")
        print(f"Sentiment: {result['predicted_label']} (confidence: {result['confidence']:.3f})")
        print("---")
