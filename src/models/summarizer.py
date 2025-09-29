# src/models/summarizer.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union
import sys

# Add project root to path for legacy compatibility
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.core.config import get_config

# Get configuration
config = get_config()


class VietnameseSummarizer:
    """
    Vietnamese Text Summarization Model
    
    This class handles:
    - Loading pre-trained Vietnamese T5 models
    - Generating summaries for Vietnamese text
    - Evaluating model performance with ROUGE metrics
    - Batch processing of texts
    """
    
    def __init__(
        self, 
        model_path: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the Vietnamese Summarizer
        
        Args:
            model_path: Path to trained model (defaults to checkpoint-2166)
            cache_dir: Cache directory for models (defaults to project cache)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        if model_path is None:
            model_path = config.summarization_model_dir / "checkpoint-2166"
        if cache_dir is None:
            cache_dir = config.summarization_cache_dir
            
        self.model_path = Path(model_path)
        self.cache_dir = Path(cache_dir)
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Flag to track if we've fallen back to CPU due to CUDA errors
        self.cuda_failed = False
            
        print(f"Using device: {self.device}")
        print(f"Loading model from: {self.model_path}")
        
        self._load_model()
        
        try:
            self.rouge = evaluate.load('rouge')
        except Exception as e:
            print(f"Warning: Could not load ROUGE metric: {e}")
            self.rouge = None
    
    def _load_model(self):
        """Load the tokenizer and model"""
        try:
            # Load tokenizer with specific configuration
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), 
                cache_dir=str(self.cache_dir)
            )
            
            # Ensure tokenizer has required special tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                str(self.model_path), 
                cache_dir=str(self.cache_dir)
            )
            
            # Try to move to device with error handling
            try:
                self.model = self.model.to(self.device)
                # Resize token embeddings if needed
                self.model.resize_token_embeddings(len(self.tokenizer))
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"CUDA error during model loading: {e}")
                    print("Falling back to CPU...")
                    self.device = "cpu"
                    self.cuda_failed = True
                    self.model = self.model.to(self.device)
                    self.model.resize_token_embeddings(len(self.tokenizer))
                else:
                    raise
            
            print("Model loaded successfully!")
            print(f"Vocab size: {len(self.tokenizer)}")
            print(f"Model vocab size: {self.model.config.vocab_size}")
            print(f"Device: {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def summarize(self, text: str) -> str:
        """
        Vietnamese text summarization with anti-hallucination controls
        
        Args:
            text: Input Vietnamese text to summarize
            
        Returns:
            Generated summary as string
        """
        if not text or len(text.strip()) == 0:
            return ""
        
        text = text.strip()
        text_length = len(text)
        
        # For very short texts, return as-is
        if text_length < 200:
            return text
        
        # Fixed generation parameters to reduce hallucination
        max_length = min(100, int(text_length * 0.3))  # 30% of original  
        min_length = max(30, int(text_length * 0.15))  # 15% of original
        
        # Conservative limits to prevent hallucination
        max_length = min(max_length, 120)  # Hard cap
        min_length = min(min_length, max_length - 10)
        
        # Simple prompt without excessive instruction
        prompted_text = f"Tóm tắt: {text}"
        
        # Tokenize with length check
        inputs = self.tokenizer(
            prompted_text,
            return_tensors="pt",
            truncation=True,
            max_length=400,  # Conservative input limit
            padding=True
        ).to(self.device)
        
        # Generate with conservative parameters to reduce hallucination
        try:
            with torch.no_grad():
                summary_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=2,  # Reduced beams for more focused generation
                    length_penalty=1.2,  # Slight penalty for length
                    repetition_penalty=1.2,  # Higher to avoid repetition
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    do_sample=False,  # Deterministic
                    temperature=1.0,  # Neutral temperature
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # Additional controls to reduce hallucination
                    forced_eos_token_id=self.tokenizer.eos_token_id,
                    suppress_tokens=None
                )
        except Exception as e:
            print(f"Generation error: {e}")
            # Ultra-conservative fallback
            with torch.no_grad():
                summary_ids = self.model.generate(
                    **inputs,
                    max_length=min(60, max_length),
                    min_length=min(20, min_length),
                    num_beams=1,  # Greedy decoding
                    early_stopping=True,
                    do_sample=False
                )
        
        # Decode and clean
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Remove prompt prefix
        if summary.startswith("Tóm tắt: "):
            summary = summary[9:].strip()
        
        # Additional cleaning to prevent hallucination artifacts
        summary = self._clean_summary(summary, text)
        
        return summary.strip()

    def _clean_summary(self, summary: str, original_text: str) -> str:
        """
        Clean and validate summary to prevent hallucination
        
        Args:
            summary: Generated summary text
            original_text: Original input text for validation
            
        Returns:
            Cleaned summary with hallucinated content removed
        """
        if not summary:
            return ""
        
        # Remove common hallucination patterns in Vietnamese
        hallucination_markers = [
            "nổi tiếng", "đặc sản", "thiên đường", "danh lam thắng cảnh",
            "đáng tham quan", "hấp dẫn du khách", "nên ghé thăm"
        ]
        
        # Basic validation: if summary contains terms not in original, be cautious
        original_lower = original_text.lower()
        summary_lower = summary.lower()
        
        # Split into sentences for better control
        sentences = [s.strip() for s in summary.split('.') if s.strip()]
        validated_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check if sentence contains hallucination markers not in original
            has_hallucination = False
            for marker in hallucination_markers:
                if marker in sentence_lower and marker not in original_lower:
                    has_hallucination = True
                    break
            
            # Keep sentence if it doesn't contain obvious hallucinations
            if not has_hallucination:
                validated_sentences.append(sentence)
            else:
                print(f"Filtered hallucinated sentence: {sentence}")
        
        # Reconstruct summary
        cleaned_summary = '. '.join(validated_sentences)
        if cleaned_summary and not cleaned_summary.endswith('.'):
            cleaned_summary += '.'
        
        # Fallback: if all sentences were filtered, use first part of original
        if not cleaned_summary or len(cleaned_summary.strip()) < 10:
            # Take first meaningful part of original text
            words = original_text.split()
            if len(words) > 20:
                cleaned_summary = ' '.join(words[:20]) + '...'
            else:
                cleaned_summary = original_text
        
        return cleaned_summary
    
    
    def should_summarize(self, text: str, min_summary_length: int = 300) -> bool:
        """
        Determine if text should be summarized or is too short
        
        Args:
            text: Input text to check
            min_summary_length: Minimum text length to warrant summarization
            
        Returns:
            True if text should be summarized, False if too short
        """
        return len(text.strip()) >= min_summary_length
    
    def smart_process(self, text: str, sentiment_analyzer=None) -> dict:
        """
        Smart pipeline that decides whether to summarize based on text length
        
        Args:
            text: Input Vietnamese text
            sentiment_analyzer: Optional sentiment analyzer instance
            
        Returns:
            Dictionary with 'summary', 'should_summarize', 'text_length', and optionally 'sentiment'
        """
        text = text.strip()
        text_length = len(text)
        should_summarize = self.should_summarize(text)
        
        result = {
            'text_length': text_length,
            'should_summarize': should_summarize,
            'original_text': text
        }
        
        if should_summarize:
            # Text is long enough - generate summary
            result['summary'] = self.summarize(text)
            result['processed_text'] = result['summary']
        else:
            # Text is too short - use original text
            result['summary'] = None
            result['processed_text'] = text
            
        # Add sentiment analysis if analyzer provided
        if sentiment_analyzer is not None:
            try:
                # Use processed_text (summary for long texts, original for short texts)
                result['sentiment'] = sentiment_analyzer.predict(result['processed_text'])
            except Exception as e:
                result['sentiment'] = f"Error: {e}"
                
        return result
    
    def summarize_batch(
        self, 
        texts: List[str],
        show_progress: bool = True
    ) -> List[str]:
        """
        Summarize multiple texts with consistent quality control
        
        Args:
            texts: List of Vietnamese texts to summarize
            show_progress: Whether to show progress bar
            
        Returns:
            List of generated summaries
        """
        if not texts:
            return []
        
        summaries = []
        iterator = texts
        if show_progress:
            iterator = tqdm(texts, desc="Generating summaries")
            
        for text in iterator:
            try:
                summary = self.summarize(text)
                summaries.append(summary)
            except Exception as e:
                print(f"Error summarizing text: {e}")
                # Fallback to truncated original
                words = text.split()
                if len(words) > 15:
                    fallback = ' '.join(words[:15]) + '...'
                else:
                    fallback = text
                summaries.append(fallback)
        
        return summaries
    
    def evaluate_on_dataset(
        self, 
        dataset_path: Optional[Union[str, Path]] = None,
        num_samples: int = 100,
        text_column: str = "Contents",
        summary_column: str = "Summary",
        **generation_kwargs
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset using ROUGE metrics
        
        Args:
            dataset_path: Path to CSV dataset (defaults to articles_clean.csv)
            num_samples: Number of samples to evaluate on
            text_column: Name of column containing input text
            summary_column: Name of column containing reference summaries
            **generation_kwargs: Additional arguments for generation
            
        Returns:
            Dictionary containing ROUGE scores and analysis
        """
        if self.rouge is None:
            raise RuntimeError("ROUGE metric not available for evaluation")
            
        # Load dataset
        if dataset_path is None:
            dataset_path = config.summarization_data
            
        print(f"Loading test data from: {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} samples for evaluation")
        
        # Sample data for evaluation
        if num_samples > len(df):
            num_samples = len(df)
            
        test_sample = df.sample(n=num_samples, random_state=42)
        
        # Generate predictions
        print(f"Evaluating on {num_samples} samples...")
        texts = test_sample[text_column].tolist()
        references = test_sample[summary_column].tolist()
        
        predictions = self.summarize_batch(texts, **generation_kwargs)
        
        # Calculate ROUGE scores
        results = self.rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=['rouge1', 'rouge2', 'rougeL']
        )
        
        # Additional analysis
        analysis = self._analyze_predictions(predictions, references)
        
        # Combine results
        evaluation_results = {
            'rouge1': results['rouge1'],
            'rouge2': results['rouge2'],
            'rougeL': results['rougeL'],
            'num_samples': num_samples,
            **analysis
        }
        
        return evaluation_results, predictions, references
    
    def _analyze_predictions(self, predictions: List[str], references: List[str]) -> Dict:
        """Analyze prediction quality"""
        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references]
        
        return {
            'avg_prediction_length': np.mean(pred_lengths),
            'avg_reference_length': np.mean(ref_lengths),
            'avg_length_ratio': np.mean([p/r if r > 0 else 0 for p, r in zip(pred_lengths, ref_lengths)]),
            'prediction_lengths': pred_lengths,
            'reference_lengths': ref_lengths
        }
    
    def find_best_worst_examples(
        self, 
        predictions: List[str], 
        references: List[str], 
        texts: List[str] = None
    ) -> Tuple[Dict, Dict]:
        """Find best and worst examples by ROUGE-1 score"""
        if self.rouge is None:
            raise RuntimeError("ROUGE metric not available")
            
        # Calculate individual ROUGE scores
        individual_scores = []
        for pred, ref in zip(predictions, references):
            score = self.rouge.compute(predictions=[pred], references=[ref])
            individual_scores.append(score['rouge1'])
        
        # Find best and worst
        best_idx = np.argmax(individual_scores)
        worst_idx = np.argmin(individual_scores)
        
        best_example = {
            'index': best_idx,
            'rouge1': individual_scores[best_idx],
            'prediction': predictions[best_idx],
            'reference': references[best_idx]
        }
        
        worst_example = {
            'index': worst_idx,
            'rouge1': individual_scores[worst_idx],
            'prediction': predictions[worst_idx],
            'reference': references[worst_idx]
        }
        
        if texts:
            best_example['text'] = texts[best_idx]
            worst_example['text'] = texts[worst_idx]
            
        return best_example, worst_example
    
    def save_model(self, output_path: Union[str, Path]):
        """Save the current model and tokenizer"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"Model saved to: {output_path}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_path': str(self.model_path),
            'cache_dir': str(self.cache_dir),
            'device': self.device,
            'model_type': self.model.config.model_type,
            'vocab_size': self.tokenizer.vocab_size,
            'max_length': self.tokenizer.model_max_length
        }


# Convenience functions for easy usage
def create_summarizer(checkpoint: str = "2166") -> VietnameseSummarizer:
    """
    Create a summarizer with default settings
    
    Args:
        checkpoint: Which checkpoint to use ("2000" or "2166")
        
    Returns:
        VietnameseSummarizer instance
    """
    if checkpoint == "2000":
        model_path = config.summarization_model_dir / "checkpoint-2000"
    else:
        model_path = config.summarization_model_dir / "checkpoint-2166"
        
    return VietnameseSummarizer(model_path=model_path)


def quick_summarize(text: str, checkpoint: str = "2166") -> str:
    """
    Quick function to summarize a single text with smart logic
    
    Args:
        text: Vietnamese text to summarize
        checkpoint: Which model checkpoint to use
        
    Returns:
        Generated summary or original text if too short
    """
    summarizer = create_summarizer(checkpoint)
    result = summarizer.smart_process(text)
    return result['processed_text']


def smart_pipeline(text: str, checkpoint: str = "2166", sentiment_analyzer=None) -> dict:
    """
    Smart pipeline that combines summarization and sentiment analysis
    
    Args:
        text: Vietnamese text to process
        checkpoint: Which model checkpoint to use
        sentiment_analyzer: Optional sentiment analyzer
        
    Returns:
        Dictionary with processing results
    """
    summarizer = create_summarizer(checkpoint)
    return summarizer.smart_process(text, sentiment_analyzer)


if __name__ == "__main__":
    # Example usage with smart pipeline
    summarizer = create_summarizer()
    
    # Test with short text
    short_text = "Hôm nay trời đẹp, tôi đi chơi với bạn."
    print("=== SHORT TEXT TEST ===")
    print(f"Input: {short_text}")
    result_short = summarizer.smart_process(short_text)
    print(f"Length: {result_short['text_length']} chars")
    print(f"Should summarize: {result_short['should_summarize']}")
    print(f"Output: {result_short['processed_text']}")
    print()
    
    # Test with long text
    long_text = """
    Hôm nay, Chủ tịch nước Nguyễn Xuân Phúc đã có cuộc gặp với Thủ tướng Chính phủ Phạm Minh Chính 
    tại Phủ Chủ tịch để bàn bạc về các vấn đề quan trọng của đất nước. Cuộc họp kéo dài 2 tiếng đồng hồ, 
    tập trung vào việc thảo luận kế hoạch phát triển kinh tế trong năm 2025. Các chuyên gia kinh tế cũng 
    được mời tham gia để đưa ra những đánh giá và khuyến nghị về tình hình kinh tế hiện tại cũng như 
    triển vọng trong tương lai. Cuộc họp được đánh giá là rất quan trọng trong việc định hướng chính sách 
    kinh tế của Việt Nam trong thời gian tới.
    """
    print("=== LONG TEXT TEST ===")
    print(f"Input: {long_text.strip()[:100]}...")
    result_long = summarizer.smart_process(long_text)
    print(f"Length: {result_long['text_length']} chars")
    print(f"Should summarize: {result_long['should_summarize']}")
    print(f"Summary: {result_long['summary']}")
    print(f"Output: {result_long['processed_text']}")
