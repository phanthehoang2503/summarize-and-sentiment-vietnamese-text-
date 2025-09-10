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

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config


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
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), 
                cache_dir=str(self.cache_dir)
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                str(self.model_path), 
                cache_dir=str(self.cache_dir)
            ).to(self.device)
            
            print("Model loaded successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def summarize(
        self, 
        text: str, 
        max_length: int = 300,
        min_length: int = None,  # Will be calculated dynamically
        num_beams: int = 6,
        length_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 2.0,
        early_stopping: bool = True
    ) -> str:
        """
        Generate summary for a single text with dynamic min_length
        
        Args:
            text: Input Vietnamese text to summarize
            max_length: Maximum length of generated summary
            min_length: Minimum length (auto-calculated if None)
            num_beams: Number of beams for beam search
            length_penalty: Length penalty for generation
            no_repeat_ngram_size: Size of n-grams that cannot repeat
            repetition_penalty: Penalty for repetition
            early_stopping: Whether to stop early when all beams reach EOS
            
        Returns:
            Generated summary as string
        """
        if not text or len(text.strip()) == 0:
            return ""
        
        # Calculate dynamic min_length based on input text length
        if min_length is None:
            text_length = len(text)
            if text_length < 200:
                # Very short text - minimal summary
                min_length = max(3, text_length // 20)
            elif text_length < 500:
                # Short text - conservative summary  
                min_length = max(8, text_length // 18)
            elif text_length < 1000:
                # Medium text - moderate summary
                min_length = max(12, text_length // 16)
            elif text_length < 2000:
                # Long text - substantial summary
                min_length = max(20, text_length // 18)
            else:
                # Very long text - comprehensive summary
                min_length = max(30, min(60, text_length // 20))
        
        # Adjust max_length to be reasonable relative to input and min_length
        if max_length < min_length * 2:
            max_length = min_length * 4
        
        # Cap max_length for aggressive compression (20-30% of original)
        input_based_max = max(min_length * 2, text_length // 4)  # 25% max
        max_length = min(max_length, input_based_max)
            
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                early_stopping=early_stopping
            )
        
        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()
    
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
