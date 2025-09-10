import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path

class VietnameseSummarizer:
    def __init__(self, model_dir=None, cache_dir=None):
        """Initialize summarizer with trained model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use latest checkpoint if model_dir not specified
        if model_dir is None:
            model_dir = Path(__file__).parent.parent / "models" / "summarizer" / "checkpoint-1854"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, cache_dir=cache_dir).to(self.device)

    def generate_summary(self, text, max_length=128, num_beams=4, temperature=0.7):
        """Generate summary from preprocessed text"""
        # Input validation
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate summary
        summary_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            length_penalty=1.5,
            no_repeat_ngram_size=3,
            repetition_penalty=2.5,
            early_stopping=True,
            do_sample=True
        )
        
        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()