#!/usr/bin/env python3
"""
Combined Summarization and Sentiment Analysis Pipeline
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path for legacy compatibility
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.summarizer import VietnameseSummarizer
from src.models.sentiment import VietnameseSentimentAnalyzer, create_sentiment_analyzer
from app.core.config import get_config

# Get configuration
config = get_config()


class SummarizationSentimentPipeline:
    """
    Combined pipeline for Vietnamese text summarization and sentiment analysis
    
    This pipeline:
    1. Takes input text (can be long)
    2. Generates a summary using ViT5 model
    3. Analyzes sentiment of the summary using PhoBERT
    4. Returns both summary and sentiment results
    """
    
    def __init__(
        self,
        summarizer_model_path: Optional[str] = None,
        sentiment_model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the combined pipeline
        
        Args:
            summarizer_model_path: Path to summarization model
            sentiment_model_path: Path to sentiment analysis model  
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        print("Initializing Summarization + Sentiment Pipeline...")
        
        try:
            print("Loading summarization model...")
            self.summarizer = VietnameseSummarizer(
                model_path=summarizer_model_path,
                device=device
            )
            print("Summarization model loaded successfully")
        except Exception as e:
            print(f"Error loading summarization model: {e}")
            raise
        
        try:
            print("Loading sentiment analysis model...")
            self.sentiment_analyzer = create_sentiment_analyzer()
            print("Sentiment analysis model loaded successfully")
        except Exception as e:
            print(f"Error loading sentiment model: {e}")
            print("Trying alternative initialization...")
            try:
                from src.models.sentiment import VietnameseSentimentAnalyzer
                self.sentiment_analyzer = VietnameseSentimentAnalyzer()
                print("Sentiment model loaded with alternative method")
            except Exception as e2:
                print(f"Alternative initialization also failed: {e2}")
                raise
        
        print("Pipeline initialized successfully!")
    
    def analyze(
        self,
        text: str,
        # Summarization parameters
        quality_mode: str = "balanced",  # Add quality_mode parameter
        max_summary_length: int = 128,
        min_summary_length: int = 10,
        summarization_params: Optional[Dict] = None,
        # Sentiment parameters
        return_sentiment_probabilities: bool = True,
        # Pipeline parameters
        min_text_length_for_summary: int = 50,  # Lowered from 200 to 50 characters
        force_summarization: bool = False,
        # Output parameters
        include_original_text: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze text with summarization and sentiment analysis
        
        Args:
            text: Input Vietnamese text to analyze
            max_summary_length: Maximum length of summary
            min_summary_length: Minimum length of summary
            summarization_params: Additional parameters for summarization
            return_sentiment_probabilities: Whether to return sentiment probabilities
            min_text_length_for_summary: Minimum text length to trigger summarization
            force_summarization: Force summarization even for short texts
            include_original_text: Whether to include original text in results
            
        Returns:
            Dictionary containing summary and sentiment analysis results
        """
        if not text or len(text.strip()) == 0:
            return {
                "error": "Empty input text",
                "original_text": text if include_original_text else None,
                "summary": "",
                "sentiment": None,
                "used_summarization": False
            }
        
        try:
            # Decide whether to use summarization based on text length
            text_length = len(text.strip())
            
            if text_length < 50:
                # For very short texts (< 50 chars), skip summarization entirely
                print(f"Text too short ({text_length} chars < 50) - performing sentiment analysis only")
                should_summarize = False
                text_for_sentiment = text
                summary = None
                used_summarization = False
            else:
                # For longer texts, check summarization threshold
                should_summarize = force_summarization or text_length >= min_text_length_for_summary
                
                if should_summarize:
                    # Step 1: Generate summary
                    print("Generating summary...")
                    
                    # Prepare summarization parameters (conservative settings)
                    sum_params = {
                        "max_length": 300,  # More conservative than before
                        "min_length": None,  # Use dynamic calculation
                        "num_beams": 3,  # Reduced for stability
                        "length_penalty": 1.0,  # Neutral
                        "no_repeat_ngram_size": 2,  # Reduced
                        "repetition_penalty": 1.1,  # Very conservative
                        "early_stopping": True,
                        "quality_mode": quality_mode  # Use the passed quality mode
                    }
                    
                    if summarization_params:
                        sum_params.update(summarization_params)
                    
                    summary = self.summarizer.summarize(text, **sum_params)
                    
                    if not summary or len(summary.strip()) == 0 or len(summary.strip()) < 10:
                        print("Summary too short or empty, using original text for sentiment analysis")
                        text_for_sentiment = text
                        summary = None
                        used_summarization = False
                    else:
                        text_for_sentiment = summary
                        used_summarization = True
                else:
                    print(f"Text length ({text_length} chars) below summarization threshold ({min_text_length_for_summary}) - analyzing sentiment directly")
                    text_for_sentiment = text
                    summary = None
                    used_summarization = False
            
            # Step 2: Analyze sentiment
            print("Analyzing sentiment...")
            try:
                sentiment_result = self.sentiment_analyzer.predict_sentiment(
                    text_for_sentiment, 
                    return_probabilities=return_sentiment_probabilities
                )
                print(f"Sentiment analysis completed: {sentiment_result.get('predicted_label', 'unknown')}")
            except Exception as e:
                print(f"Sentiment analysis failed: {e}")
                # Fallback sentiment result
                sentiment_result = {
                    'predicted_label': 'UNKNOWN',
                    'confidence': 0.0,
                    'probabilities': {'POS': 0.33, 'NEU': 0.34, 'NEG': 0.33},
                    'error': str(e)
                }
            
            # Prepare result
            result = {
                "success": True,
                "original_text": text if include_original_text else None,
                "original_length": len(text),
                "summary": summary if used_summarization else None,
                "summary_length": len(summary) if summary and used_summarization else None,
                "compression_ratio": len(summary) / len(text) if summary and used_summarization and len(text) > 0 else None,
                "text_analyzed_for_sentiment": text_for_sentiment,
                "sentiment": sentiment_result,
                "used_summarization": used_summarization,
                "analysis_method": "summarization + sentiment" if used_summarization else "direct sentiment only",
                "text_length": len(text.strip()),
                "skip_reason": "text_too_short" if len(text.strip()) < 50 else None
            }
            
            return result
            
        except Exception as e:
            return {
                "error": f"Pipeline error: {str(e)}",
                "original_text": text if include_original_text else None,
                "summary": "",
                "sentiment": None,
                "used_summarization": False
            }
    
    def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 4,
        show_progress: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts with summarization and sentiment analysis
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            **kwargs: Additional parameters for analyze()
            
        Returns:
            List of analysis results
        """
        results = []
        
        if show_progress:
            from tqdm import tqdm
            texts_iter = tqdm(texts, desc="Processing texts")
        else:
            texts_iter = texts
        
        for text in texts_iter:
            result = self.analyze(text, **kwargs)
            results.append(result)
        
        return results
    
    def analyze_document(
        self,
        text: str,
        chunk_size: int = 2000,
        overlap: int = 200,
        combine_summaries: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze very long documents by chunking them
        
        Args:
            text: Long input text
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
            combine_summaries: Whether to combine chunk summaries
            **kwargs: Additional parameters for analyze()
            
        Returns:
            Analysis results for the document
        """
        if len(text) <= chunk_size:
            return self.analyze(text, **kwargs)
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                for i in range(end - 50, min(end + 50, len(text))):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunks.append(text[start:end])
            start = end - overlap if end < len(text) else end
        
        print(f"Split document into {len(chunks)} chunks")
        
        # Analyze each chunk
        chunk_results = self.analyze_batch(
            chunks, 
            show_progress=True,
            **{k: v for k, v in kwargs.items() if k != 'include_original_text'}
        )
        
        # Combine results
        if combine_summaries:
            # Combine all summaries
            all_summaries = [r["summary"] for r in chunk_results if r.get("summary")]
            combined_summary_text = " ".join(all_summaries)
            
            # Analyze sentiment of combined summary
            if combined_summary_text:
                combined_sentiment = self.sentiment_analyzer.predict_sentiment(
                    combined_summary_text,
                    return_probabilities=kwargs.get("return_sentiment_probabilities", True)
                )
            else:
                combined_sentiment = None
            
            return {
                "success": True,
                "original_text": text if kwargs.get("include_original_text", False) else None,
                "original_length": len(text),
                "num_chunks": len(chunks),
                "chunk_results": chunk_results,
                "combined_summary": combined_summary_text,
                "combined_summary_length": len(combined_summary_text) if combined_summary_text else 0,
                "compression_ratio": len(combined_summary_text) / len(text) if combined_summary_text and len(text) > 0 else 0,
                "sentiment": combined_sentiment
            }
        else:
            return {
                "success": True,
                "original_text": text if kwargs.get("include_original_text", False) else None,
                "original_length": len(text),
                "num_chunks": len(chunks),
                "chunk_results": chunk_results
            }


def create_pipeline(
    summarizer_model_path: Optional[str] = None,
    sentiment_model_path: Optional[str] = None,
    device: Optional[str] = None
) -> SummarizationSentimentPipeline:
    """
    Create a new summarization + sentiment pipeline
    
    Args:
        summarizer_model_path: Path to summarization model
        sentiment_model_path: Path to sentiment analysis model
        device: Device to use
        
    Returns:
        Initialized pipeline
    """
    return SummarizationSentimentPipeline(
        summarizer_model_path=summarizer_model_path,
        sentiment_model_path=sentiment_model_path,
        device=device
    )


if __name__ == "__main__":
    # Quick test
    pipeline = create_pipeline()
    
    test_text = """
    Hôm nay tôi đã có một trải nghiệm tuyệt vời tại nhà hàng mới. 
    Món ăn rất ngon, dịch vụ chu đáo và không gian rất đẹp. 
    Nhân viên phục vụ rất nhiệt tình và chuyên nghiệp. 
    Giá cả cũng hợp lý so với chất lượng. 
    Tôi chắc chắn sẽ quay lại đây lần nữa và giới thiệu cho bạn bè.
    """
    
    result = pipeline.analyze(test_text, include_original_text=True)
    
    print("\nPipeline Test Results:")
    print("=" * 50)
    print(f"Original length: {result['original_length']} characters")
    print(f"Summary: {result['summary']}")
    print(f"Summary length: {result['summary_length']} characters")
    print(f"Compression ratio: {result['compression_ratio']:.2f}")
    print(f"Sentiment: {result['sentiment']['predicted_label']}")
    print(f"Confidence: {result['sentiment']['confidence']:.3f}")
