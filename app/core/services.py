"""
Text Analysis Service Layer
Business logic for Vietnamese text summarization and sentiment analysis
"""
from typing import Dict, Any, Optional
import time
import logging
import threading
from pathlib import Path

# Import performance utilities
from app.core.performance import (
    cache_api_response, 
    api_response_cache, 
    performance_tracker,
    get_text_cache_key
)

# Import models with proper error handling
try:
    from src.models.sentiment import create_sentiment_analyzer
    from src.models.summarizer import create_summarizer  
    from src.models.pipeline import SummarizationSentimentPipeline
except ImportError as e:
    logging.error(f"Failed to import models: {e}")
    raise


class TextAnalysisService:
    """
    Service class with thread-safe singleton model loading for demo use.
    Provides a clean interface for the web layer.
    """
    
    def __init__(self):
        """Initialize the text analysis service with thread-safe lazy loading"""
        self._sentiment_analyzer = None
        self._summarizer = None
        self._pipeline = None
        self._lock = threading.Lock()  # For thread-safe initialization
        self.logger = logging.getLogger(__name__)
        self._initialization_errors = {}
    
    @property
    def sentiment_analyzer(self):
        """Thread-safe lazy-loaded sentiment analyzer"""
        if self._sentiment_analyzer is None:
            with self._lock:
                if self._sentiment_analyzer is None:  # Double-check pattern
                    try:
                        self.logger.info("Initializing sentiment analyzer...")
                        self._sentiment_analyzer = create_sentiment_analyzer()
                        self.logger.info("Sentiment analyzer loaded successfully")
                    except Exception as e:
                        self.logger.error(f"Failed to load sentiment analyzer: {e}")
                        self._initialization_errors['sentiment'] = str(e)
                        raise
        return self._sentiment_analyzer
    
    @property
    def summarizer(self):
        """Thread-safe lazy-loaded summarizer"""
        if self._summarizer is None:
            with self._lock:
                if self._summarizer is None:  # Double-check pattern
                    try:
                        self.logger.info("Initializing summarizer...")
                        self._summarizer = create_summarizer()
                        self.logger.info("Summarizer loaded successfully")
                    except Exception as e:
                        self.logger.error(f"Failed to load summarizer: {e}")
                        self._initialization_errors['summarizer'] = str(e)
                        raise
        return self._summarizer
    
    @property
    def pipeline(self):
        """Thread-safe lazy-loaded combined pipeline"""
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:  # Double-check pattern
                    try:
                        self.logger.info("Initializing combined pipeline...")
                        self._pipeline = SummarizationSentimentPipeline()
                        self.logger.info("Combined pipeline loaded successfully")
                    except Exception as e:
                        self.logger.error(f"Failed to load pipeline: {e}")
                        self._initialization_errors['pipeline'] = str(e)
                        raise
        return self._pipeline
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services for health checks"""
        status = {
            'services': {
                'sentiment_analyzer': 'not_loaded',
                'summarizer': 'not_loaded', 
                'pipeline': 'not_loaded'
            },
            'errors': self._initialization_errors.copy()
        }
        
        # Check what's currently loaded
        if self._sentiment_analyzer is not None:
            status['services']['sentiment_analyzer'] = 'loaded'
        if self._summarizer is not None:
            status['services']['summarizer'] = 'loaded'
        if self._pipeline is not None:
            status['services']['pipeline'] = 'loaded'
            
        return status
    
    def is_vietnamese_text(self, text: str) -> bool:
        """Simple Vietnamese text detection"""
        vietnamese_chars = 'àáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽềềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ'
        vietnamese_chars += vietnamese_chars.upper()
        
        vietnamese_count = sum(1 for char in text if char in vietnamese_chars)
        total_chars = len([char for char in text if char.isalpha()])
        
        if total_chars == 0:
            return False
        
        return (vietnamese_count / total_chars) > 0.1  # At least 10% Vietnamese characters
    
    def count_tokens(self, text: str) -> int:
        """Simple token counting"""
        return len(text.split())
    
    @cache_api_response()
    def summarize_text(self, text: str) -> Dict[str, Any]:
        """
        Generate summary for Vietnamese text
        
        Args:
            text: Input Vietnamese text
            
        Returns:
            Dictionary with summary results and metadata
        """
        start_time = time.time()
        
        try:
            if not text or not text.strip():
                raise ValueError("Text is required")
            
            # Check if text is Vietnamese
            is_vietnamese = self.is_vietnamese_text(text)
            
            # Generate summary with default settings
            summary = self.summarizer.summarize(text)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'result': {
                    'summary': summary,
                    'original_length': len(text),
                    'summary_length': len(summary),
                    'original_tokens': self.count_tokens(text),
                    'summary_tokens': self.count_tokens(summary),
                    'compression_ratio': len(summary) / len(text) if len(text) > 0 else 0,
                    'processing_time': round(processing_time, 3),
                    'is_vietnamese': is_vietnamese
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Summarization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': round(processing_time, 3)
            }
    
    @cache_api_response()
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of Vietnamese text
        Use the same pipeline for consistency
        
        Args:
            text: Input Vietnamese text
            
        Returns:
            Dictionary with sentiment analysis results and metadata
        """
        start_time = time.time()
        
        try:
            if not text or not text.strip():
                raise ValueError("Text is required")
            
            # Check if text is Vietnamese
            is_vietnamese = self.is_vietnamese_text(text)
            
            # Use pipeline for sentiment analysis to ensure consistency
            result = self.pipeline.analyze(text, include_original_text=False)
            
            # Check if pipeline returned an error
            if 'error' in result:
                processing_time = time.time() - start_time
                return {
                    'success': False,
                    'error': result['error'],
                    'processing_time': round(processing_time, 3)
                }
            
            # Extract sentiment data from pipeline result
            sentiment_data = result.get('sentiment', {})
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'result': {
                    **sentiment_data,
                    'tokens': self.count_tokens(text),
                    'processing_time': round(processing_time, 3),
                    'is_vietnamese': is_vietnamese
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Sentiment analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': round(processing_time, 3)
            }
    
    @cache_api_response()
    def analyze_combined(self, text: str) -> Dict[str, Any]:
        """
        Perform combined summarization and sentiment analysis
        
        Args:
            text: Input Vietnamese text
            
        Returns:
            Dictionary with both summary and sentiment results
        """
        start_time = time.time()
        
        try:
            if not text or not text.strip():
                raise ValueError("Text is required")
            
            # Check if text is Vietnamese
            is_vietnamese = self.is_vietnamese_text(text)
            
            # Run combined pipeline with default settings
            result = self.pipeline.analyze(text)
            
            # Check if pipeline returned an error
            if 'error' in result:
                processing_time = time.time() - start_time
                return {
                    'success': False,
                    'error': result['error'],
                    'processing_time': round(processing_time, 3)
                }
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'result': {
                    **result,
                    'original_tokens': self.count_tokens(text),
                    'summary_tokens': self.count_tokens(result.get('summary', '')) if result.get('summary') else 0,
                    'processing_time': round(processing_time, 3),
                    'is_vietnamese': is_vietnamese
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Combined analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': round(processing_time, 3)
            }
    
    def process_file_upload(self, file_content: str, filename: str) -> Dict[str, Any]:
        """
        Process uploaded file content
        
        Args:
            file_content: Content of the uploaded file
            filename: Name of the uploaded file
            
        Returns:
            Dictionary with file processing results
        """
        try:
            if not file_content or not file_content.strip():
                raise ValueError("File is empty")
            
            return {
                'success': True,
                'content': file_content.strip(),
                'filename': filename,
                'size': len(file_content),
                'tokens': self.count_tokens(file_content)
            }
            
        except Exception as e:
            self.logger.error(f"File processing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance and caching statistics"""
        return {
            'cache_stats': api_response_cache.stats(),
            'performance_stats': performance_tracker.get_stats(),
            'service_status': self.get_service_status()
        }
    
    def clear_caches(self) -> Dict[str, str]:
        """Clear all caches for memory cleanup"""
        api_response_cache.clear()
        
        # Clear model-level caches if they exist
        if self._sentiment_analyzer and hasattr(self._sentiment_analyzer, '_tokenization_cache'):
            self._sentiment_analyzer._tokenization_cache.clear()
        
        if self._summarizer and hasattr(self._summarizer, '_tokenization_cache'):
            self._summarizer._tokenization_cache.clear()
            
        return {'status': 'Caches cleared successfully'}


# Global service instance
text_analysis_service = TextAnalysisService()