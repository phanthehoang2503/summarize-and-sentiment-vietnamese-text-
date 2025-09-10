# scripts/evaluate_model.py
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from tqdm import tqdm
import torch
from pathlib import Path

class SummarizationEvaluator:
    def __init__(self, model_path, cache_dir="D:\Project\MajorProject\cache"):
        """Initialize the evaluator with trained model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, cache_dir=cache_dir).to(self.device)
        
        # Load ROUGE metric
        self.rouge = evaluate.load('rouge')
        
    def generate_summary(self, text, max_length=128, num_beams=4):
        """Generate summary for a single text"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=1.5,
                no_repeat_ngram_size=3,
                repetition_penalty=2.5,
                early_stopping=True
            )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()
    
    def evaluate_model(self, test_df, sample_size=100):
        """Evaluate model on test dataset"""
        # Sample data for evaluation
        test_sample = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)
        
        print(f"Evaluating on {len(test_sample)} samples...")
        
        predictions = []
        references = []
        
        for idx, row in tqdm(test_sample.iterrows(), total=len(test_sample)):
            try:
                # Generate prediction
                pred_summary = self.generate_summary(row['Contents'])
                predictions.append(pred_summary)
                references.append(row['Summary'])
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        # Calculate ROUGE scores
        rouge_scores = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        
        return {
            'rouge_scores': rouge_scores,
            'predictions': predictions,
            'references': references,
            'sample_count': len(predictions)
        }
    
    def detailed_analysis(self, predictions, references):
        """Perform detailed analysis of predictions"""
        analysis = {
            'avg_prediction_length': np.mean([len(p.split()) for p in predictions]),
            'avg_reference_length': np.mean([len(r.split()) for r in references]),
            'length_ratio': [],
            'quality_samples': []
        }
        
        for pred, ref in zip(predictions, references):
            pred_len = len(pred.split())
            ref_len = len(ref.split())
            if ref_len > 0:
                analysis['length_ratio'].append(pred_len / ref_len)
        
        rouge_scores = []
        for pred, ref in zip(predictions, references):
            score = self.rouge.compute(predictions=[pred], references=[ref])
            rouge_scores.append(score['rouge1'])
        
        best_idx = np.argmax(rouge_scores)
        worst_idx = np.argmin(rouge_scores)
        
        analysis['best_example'] = {
            'prediction': predictions[best_idx],
            'reference': references[best_idx],
            'rouge1': rouge_scores[best_idx]
        }
        
        analysis['worst_example'] = {
            'prediction': predictions[worst_idx],
            'reference': references[worst_idx],
            'rouge1': rouge_scores[worst_idx]
        }
        
        return analysis

def main():
    from datasets import load_dataset
    test_data = load_dataset('csv', data_files="../data/processed/articles_clean.csv")
    test_df = pd.DataFrame(test_data["train"])
    
    model_path = "../models/summarizer/checkpoint-1854"
    evaluator = SummarizationEvaluator(model_path)
    
    # Run evaluation
    results = evaluator.evaluate_model(test_df, sample_size=50)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Samples evaluated: {results['sample_count']}")
    print(f"ROUGE-1: {results['rouge_scores']['rouge1']:.4f}")
    print(f"ROUGE-2: {results['rouge_scores']['rouge2']:.4f}")
    print(f"ROUGE-L: {results['rouge_scores']['rougeL']:.4f}")
    
    # Detailed analysis
    analysis = evaluator.detailed_analysis(results['predictions'], results['references'])
    print(f"\nDetailed Analysis:")
    print(f"Average prediction length: {analysis['avg_prediction_length']:.1f} words")
    print(f"Average reference length: {analysis['avg_reference_length']:.1f} words")
    print(f"Average length ratio: {np.mean(analysis['length_ratio']):.3f}")
    
    print(f"\nBest Example (ROUGE-1: {analysis['best_example']['rouge1']:.4f}):")
    print(f"Prediction: {analysis['best_example']['prediction']}")
    print(f"Reference: {analysis['best_example']['reference']}")
    
    print(f"\nWorst Example (ROUGE-1: {analysis['worst_example']['rouge1']:.4f}):")
    print(f"Prediction: {analysis['worst_example']['prediction']}")
    print(f"Reference: {analysis['worst_example']['reference']}")

if __name__ == "__main__":
    main()
