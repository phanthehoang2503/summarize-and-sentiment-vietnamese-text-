"""
Text Analysis Service Layer
Business logic for Vietnamese text summarization and sentiment analysis
"""
from typing import Dict, Any, Optional
import time
import logging
from pathlib import Path

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
    Service class that encapsulates all text analysis business logic.
    Provides a clean interface for the web layer.
    """
    
    def __init__(self):
        """Initialize the text analysis service with lazy loading"""
        self._sentiment_analyzer = None
        self._summarizer = None
        self._pipeline = None
        self.logger = logging.getLogger(__name__)
    
    @property
    def sentiment_analyzer(self):
        """Lazy-loaded sentiment analyzer"""
        if self._sentiment_analyzer is None:
            self._sentiment_analyzer = create_sentiment_analyzer()
        return self._sentiment_analyzer
    
    @property
    def summarizer(self):
        """Lazy-loaded summarizer"""
        if self._summarizer is None:
            self._summarizer = create_summarizer()
        return self._summarizer
    
    @property
    def pipeline(self):
        """Lazy-loaded combined pipeline"""
        if self._pipeline is None:
            self._pipeline = SummarizationSentimentPipeline()
        return self._pipeline
    
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
    
    def summarize_text(self, text: str, quality_mode: str = "balanced") -> Dict[str, Any]:
        """
        Generate summary for Vietnamese text
        
        Args:
            text: Input Vietnamese text
            quality_mode: Summary quality ("balanced", "detailed")
            
        Returns:
            Dictionary with summary results and metadata
        """
        start_time = time.time()
        
        try:
            if not text or not text.strip():
                raise ValueError("Text is required")
            
            # Validate quality mode
            valid_modes = ["balanced", "detailed"]
            if quality_mode not in valid_modes:
                quality_mode = "balanced"
            
            # Check if text is Vietnamese
            is_vietnamese = self.is_vietnamese_text(text)
            
            # Generate summary with quality mode
            summary = self.summarizer.summarize(text, quality_mode=quality_mode)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'result': {
                    'summary': summary,
                    'quality_mode': quality_mode,
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
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of Vietnamese text
        
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
            
            # Analyze sentiment
            result = self.sentiment_analyzer.predict_sentiment(text, return_probabilities=True)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'result': {
                    **result,
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
    
    def analyze_combined(self, text: str, quality_mode: str = "balanced") -> Dict[str, Any]:
        """
        Perform combined summarization and sentiment analysis
        
        Args:
            text: Input Vietnamese text
            quality_mode: Summary quality ("balanced", "detailed")
            
        Returns:
            Dictionary with both summary and sentiment results
        """
        start_time = time.time()
        
        try:
            if not text or not text.strip():
                raise ValueError("Text is required")
            
            # Validate quality mode
            valid_modes = ["balanced", "detailed"]
            if quality_mode not in valid_modes:
                quality_mode = "balanced"
            
            # Check if text is Vietnamese
            is_vietnamese = self.is_vietnamese_text(text)
            
            # Run combined pipeline with quality mode
            result = self.pipeline.analyze(text, quality_mode=quality_mode)
            
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


# Global service instance
text_analysis_service = TextAnalysisService()