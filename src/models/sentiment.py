"""
Vietnamese Sentiment Analysis Model
Clean implementation with robust error handling
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import logging
from typing import Dict, Optional, Union, List
import warnings

from app.core.config import get_config

# Configure logging
logger = logging.getLogger(__name__)

# Suppress transformer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Get configuration
config = get_config()


class VietnameseSentimentAnalyzer:
    """Vietnamese Sentiment Analysis using fine-tuned PhoBERT"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None):
        """Initialize sentiment analyzer with robust device handling"""
        
        self.model_path = Path(model_path or config.sentiment_model_dir)
        self.cache_dir = Path(cache_dir or config.sentiment_cache_dir)
        
        # Device setup with fallback
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        
        # Sentiment labels - clear and consistent
        self.label_mapping = {0: "NEG", 1: "NEU", 2: "POS"}
        
        logger.info(f"SentimentAnalyzer initialized - Device: {self.device}, Model: {self.model_path}")
        
    def _validate_model_path(self) -> bool:
        """Validate model files exist"""
        required_files = ['config.json']
        # Check for either pytorch_model.bin or model.safetensors
        model_files = ['pytorch_model.bin', 'model.safetensors']
        has_model_file = any((self.model_path / f).exists() for f in model_files)
        
        for file in required_files:
            if not (self.model_path / file).exists():
                logger.error(f"Missing model file: {file}")
                return False
                
        if not has_model_file:
            logger.error(f"Missing model weights file (pytorch_model.bin or model.safetensors)")
            return False
            
        return True
        
    def load_model(self) -> bool:
        """Load model with comprehensive error handling"""
        if self._model_loaded:
            return True
            
        try:
            if not self._validate_model_path():
                raise FileNotFoundError(f"Model files not found at {self.model_path}")
                
            logger.info("Loading sentiment model...")
            
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                cache_dir=self.cache_dir,
                local_files_only=True
            )
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with device-aware error handling
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                cache_dir=self.cache_dir,
                local_files_only=True
            )
            
            # Move to device with fallback
            try:
                self.model.to(self.device)
                self.model.resize_token_embeddings(len(self.tokenizer))
            except RuntimeError as e:
                if "CUDA" in str(e) and self.device.type == "cuda":
                    logger.warning(f"CUDA error: {e}. Falling back to CPU")
                    self.device = torch.device('cpu')
                    self.model.to(self.device)
                    self.model.resize_token_embeddings(len(self.tokenizer))
                else:
                    raise
                    
            self.model.eval()
            self._model_loaded = True
            
            logger.info(f"Sentiment model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            return False
            
    def _validate_text_input(self, text: str) -> bool:
        """Validate input text"""
        if not isinstance(text, str):
            return False
        if not text.strip():
            return False
        if len(text) > 10000:  # Reasonable limit
            return False
        return True
        
    def predict_sentiment(self, 
                         text: str, 
                         return_probabilities: bool = False) -> Union[str, Dict, None]:
        """
        Predict sentiment for text with robust error handling
        
        Args:
            text: Input Vietnamese text
            return_probabilities: If True, return detailed results
            
        Returns:
            Sentiment label (str) or detailed dict, None on error
        """
        if not self._validate_text_input(text):
            logger.warning("Invalid input text for sentiment analysis")
            return None
            
        if not self._model_loaded and not self.load_model():
            logger.error("Model not loaded and failed to load")
            return None
            
        try:
            # Tokenize with safe parameters
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
                    'confidence': probabilities[0][predicted_class].item(),
                    'probabilities': {
                        label: probabilities[0][idx].item() 
                        for idx, label in self.label_mapping.items()
                    }
                }
            else:
                return self.label_mapping[predicted_class]
                
        except Exception as e:
            logger.error(f"Sentiment prediction error: {e}")
            return None
            
    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[Optional[str]]:
        """Analyze sentiment for multiple texts efficiently"""
        if not self._model_loaded and not self.load_model():
            return [None] * len(texts)
            
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch:
                result = self.predict_sentiment(text)
                batch_results.append(result)
                
            results.extend(batch_results)
            
        return results


def create_sentiment_analyzer(**kwargs) -> VietnameseSentimentAnalyzer:
    """
    Factory function to create a sentiment analyzer instance
    
    Args:
        **kwargs: Optional arguments to pass to VietnameseSentimentAnalyzer
        
    Returns:
        Configured VietnameseSentimentAnalyzer instance
    """
    return VietnameseSentimentAnalyzer(**kwargs)


def quick_sentiment_analysis(text: str) -> Optional[str]:
    """
    Quick sentiment analysis for a single text
    
    Args:
        text: Input Vietnamese text
        
    Returns:
        Sentiment label (POS/NEG/NEU) or None if analysis fails
    """
    try:
        analyzer = create_sentiment_analyzer()
        return analyzer.predict_sentiment(text)
    except Exception as e:
        logger.error(f"Quick sentiment analysis failed: {e}")
        return None