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
    
    def summarize(
        self, 
        text: str, 
        max_length: int = None,  # Will use config default if None
        min_length: int = None,  # Will be calculated dynamically
        num_beams: int = None,  # Will use config default if None
        length_penalty: float = None,  # Will use config default if None
        no_repeat_ngram_size: int = None,
        repetition_penalty: float = None,  # Will use config default if None
        early_stopping: bool = True,
        quality_mode: str = None  # "concise", "balanced", "detailed"
    ) -> str:
        """
        Generate summary for a single text with configurable quality settings
        
        Args:
            text: Input Vietnamese text to summarize
            max_length: Maximum length of generated summary
            min_length: Minimum length (auto-calculated if None)
            num_beams: Number of beams for beam search
            length_penalty: Length penalty for generation
            no_repeat_ngram_size: Size of n-grams that cannot repeat
            repetition_penalty: Penalty for repetition
            early_stopping: Whether to stop early when all beams reach EOS
            quality_mode: Quality preset ("concise", "balanced", "detailed")
            
        Returns:
            Generated summary as string
        """
        if not text or len(text.strip()) == 0:
            return ""
        
        # Add Vietnamese summarization prompt
        prompted_text = f"Tóm tắt: {text.strip()}"
        
        # Get generation settings from config
        gen_config = config.generation['summarization']
        
        # Set defaults from config if not provided, with mode-specific overrides
        if max_length is None:
            max_length = 200 if quality_mode == "detailed" else 100  # Reasonable limits
        if num_beams is None:
            num_beams = 3  # Keep consistent for quality
        if length_penalty is None:
            length_penalty = 0.9 if quality_mode == "detailed" else 0.7  # Favor shorter for balanced
        if repetition_penalty is None:
            repetition_penalty = 1.1  # Slightly higher to avoid repetition
        if no_repeat_ngram_size is None:
            no_repeat_ngram_size = 3  # Increase to avoid repetition
        
        # Apply quality mode (simplified)
        if quality_mode is None:
            quality_mode = "balanced"
        
        # Calculate text length for later use
        text_length = len(text)
        
        # Calculate dynamic min_length based on quality mode and text length  
        if min_length is None:
            if quality_mode == "balanced":
                # Aim for 5-15% of original length (concise)
                min_length = max(15, int(text_length * 0.05))
                
            elif quality_mode == "detailed":
                # Aim for 20-40% of original length (more comprehensive)
                min_length = max(30, int(text_length * 0.15))
                
            else:
                # Fallback to balanced
                min_length = max(15, int(text_length * 0.05))
        
        # Ensure min_length is an integer
        min_length = int(min_length)
        
        # Calculate max_length based on compression ratio
        if quality_mode == "balanced":
            max_length = min(max_length, max(min_length + 20, int(text_length * 0.25)))  # 25% max
        elif quality_mode == "detailed":
            max_length = min(max_length, max(min_length + 30, int(text_length * 0.50)))  # 50% max
        else:
            max_length = min(max_length, max(min_length + 20, int(text_length * 0.25)))
        
        # Ensure max_length is reasonable and an integer
        max_length = int(max_length)
        max_length = max(max_length, min_length + 10)  # At least 10 chars more than min
        
        # IMPORTANT: Ensure lengths don't exceed model limits
        # ViT5 typically has a max length of 512 tokens
        max_length = min(max_length, 256)  # Conservative limit for generation
        min_length = min(min_length, max_length - 5)  # Ensure min < max
        
        # Ensure we have reasonable bounds
        if min_length >= max_length:
            min_length = max(1, max_length - 10)
            
        print(f"Generation params: min_length={min_length}, max_length={max_length}")
            
        # Tokenize input with prompt
        inputs = self.tokenizer(
            prompted_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Check input token length to avoid issues
        input_length = inputs['input_ids'].shape[1]
        print(f"Input tokens: {input_length}")
        
        # Adjust max_length if input is very long
        if input_length > 400:
            max_length = min(max_length, 128)  # More conservative for long inputs
            min_length = min(min_length, max_length - 5)
        
        # Generate summary with error handling
        try:
            with torch.no_grad():
                summary_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    repetition_penalty=repetition_penalty,
                    early_stopping=early_stopping,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False  # Use deterministic generation
                )
        except RuntimeError as e:
            if "CUDA" in str(e) and self.device == "cuda" and not self.cuda_failed:
                print(f"CUDA error encountered: {e}")
                print("Falling back to CPU...")
                # Move model to CPU and retry
                self.model = self.model.cpu()
                self.device = "cpu"
                self.cuda_failed = True
                inputs = {k: v.cpu() for k, v in inputs.items()}
                
                with torch.no_grad():
                    summary_ids = self.model.generate(
                        **inputs,
                        max_length=min(64, max_length),
                        min_length=min(10, min_length),
                        num_beams=2,
                        length_penalty=1.0,
                        early_stopping=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        do_sample=False
                    )
            else:
                print(f"Generation error: {e}")
                # Fallback to very conservative settings
                with torch.no_grad():
                    summary_ids = self.model.generate(
                        **inputs,
                        max_length=min(64, max_length),
                        min_length=min(10, min_length),
                        num_beams=2,
                        length_penalty=1.0,
                        early_stopping=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        do_sample=False
                    )
        
        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Clean up the summary - remove the prompt if it's included in output
        if summary.startswith("Tóm tắt: "):
            summary = summary[9:]  # Remove "Tóm tắt: " prefix
        
        # Additional cleanup - remove any text that looks like it's repeating the prompt
        summary = summary.strip()
        
        # Apply quality-aware post-processing only if summary is problematic
        if len(summary) >= len(text):
            # Force summarization respecting quality mode
            sentences = summary.split('. ')
            if len(sentences) > 1:
                if quality_mode == "detailed":
                    # Take first 2-3 sentences for detailed mode
                    num_sentences = min(3, len(sentences))
                    summary = '. '.join(sentences[:num_sentences])
                    if not summary.endswith('.'):
                        summary += '.'
                else:
                    # Take first sentence for balanced mode
                    summary = sentences[0] + '.'
            else:
                # Fallback: respect quality mode ratios
                if quality_mode == "detailed":
                    summary = summary[:int(len(text) * 0.7)]  # 70% for detailed
                else:
                    summary = summary[:int(len(text) * 0.5)]  # 50% for balanced
                
                if not summary.endswith('.'):
                    summary += '.'
        
        return summary.strip()
    
    def summarize_balanced(self, text: str) -> str:
        """Generate a balanced summary (40% compression) - default"""
        return self.summarize(text, quality_mode="balanced")
    
    def summarize_detailed(self, text: str) -> str:
        """Generate a detailed summary (70% compression)"""
        return self.summarize(text, quality_mode="detailed")
    
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
        batch_size: int = 8,
        show_progress: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate summaries for multiple texts
        
        Args:
            texts: List of Vietnamese texts to summarize
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar
            **kwargs: Additional arguments for summarize()
            
        Returns:
            List of generated summaries
        """
        summaries = []
        
        # Process in batches
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating summaries")
            
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            batch_summaries = [self.summarize(text, **kwargs) for text in batch_texts]
            summaries.extend(batch_summaries)
            
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
