#!/usr/bin/env python3
"""
Demo script for Vietnamese Sentiment Analysis
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.sentiment import VietnameseSentimentAnalyzer, create_sentiment_analyzer
from config import config

def demo_single_prediction():
    """Demo single text sentiment prediction"""
    print("Single Text Sentiment Analysis Demo")
    print("=" * 50)
    
    # Create analyzer
    analyzer = create_sentiment_analyzer()
    
    # Test texts
    test_texts = [
        "Tôi rất hài lòng với sản phẩm này, chất lượng tuyệt vời!",
        "Dịch vụ khách hàng tệ quá, tôi rất thất vọng.",
        "Món ăn ngon, giá cả hợp lý, nhân viên thân thiện.",
        "Chất lượng sản phẩm không như mong đợi, rất tệ.",
        "Phim hay, diễn viên diễn xuất tốt, cốt truyện hấp dẫn.",
        "Giao hàng chậm, đóng gói không cẩn thận."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nText {i}: {text}")
        
        # Get detailed prediction
        result = analyzer.predict_sentiment(text, return_probabilities=True)
        
        print(f"   Sentiment: {result['predicted_label']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"     {label}: {prob:.3f}")

def demo_batch_prediction():
    """Demo batch sentiment prediction"""
    print("\n\nBatch Sentiment Analysis Demo")
    print("=" * 50)
    
    analyzer = create_sentiment_analyzer()
    
    # Batch of texts
    batch_texts = [
        "Sản phẩm tuyệt vời, tôi sẽ mua lại.",
        "Không hài lòng với chất lượng.",
        "Dịch vụ tốt, nhân viên nhiệt tình.",
        "Giá cả quá đắt so với chất lượng.",
        "Rất đáng tiền, chất lượng ổn định.",
        "Giao hàng nhanh, đóng gói cẩn thận.",
        "Không giống như mô tả, thất vọng.",
        "Sẽ giới thiệu cho bạn bè sử dụng."
    ]
    
    print(f"Analyzing {len(batch_texts)} texts...")
    
    results = analyzer.predict_batch(batch_texts, return_probabilities=True)
    
    print("\nResults:")
    for text, result in zip(batch_texts, results):
        sentiment = result['predicted_label']
        confidence = result['confidence']
        print(f"{sentiment.upper()} ({confidence:.3f}): {text}")

def demo_dataset_analysis():
    """Demo dataset sentiment analysis"""
    print("\n\nDataset Analysis Demo")
    print("=" * 50)
    
    try:
        # Create analyzer
        analyzer = create_sentiment_analyzer()
        
        # Analyze a sample of the dataset
        print("Analyzing sample from sentiment dataset...")
        df_results = analyzer.analyze_dataset(
            dataset_path=config.sentiment_data,
            text_column="comment",
            sample_size=10,
            batch_size=4
        )
        
        print(f"\nAnalysis Results ({len(df_results)} samples):")
        print(df_results[['comment', 'predicted_sentiment', 'confidence']].head())
        
        # Summary statistics
        sentiment_counts = df_results['predicted_sentiment'].value_counts()
        avg_confidence = df_results['confidence'].mean()
        
        print(f"\nSummary:")
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   Sentiment distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df_results)) * 100
            print(f"     {sentiment}: {count} ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"❌ Dataset analysis failed: {e}")
        print("Make sure the sentiment dataset exists and is properly formatted.")

def interactive_demo():
    """Interactive sentiment analysis"""
    print("\n\nInteractive Sentiment Analysis")
    print("=" * 50)
    print("Enter Vietnamese text to analyze sentiment (type 'quit' to exit)")
    
    analyzer = create_sentiment_analyzer()
    
    while True:
        try:
            user_input = input("\nEnter text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not user_input:
                print("Please enter some text.")
                continue
                
            # Analyze sentiment
            result = analyzer.predict_sentiment(user_input, return_probabilities=True)
            
            # Display result
            sentiment = result['predicted_label']
            confidence = result['confidence']
            
            print(f"\nResult:")
            print(f"   Sentiment: {sentiment.upper()}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Probabilities:")
            for label, prob in result['probabilities'].items():
                bar_length = int(prob * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"     {label:8s}: {bar} {prob:.3f}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main demo function"""
    print("Vietnamese Sentiment Analysis Demo")
    print("=" * 60)
    
    # Run demos
    demo_single_prediction()
    demo_batch_prediction()
    demo_dataset_analysis()
    
    # Ask if user wants interactive demo
    print("\n" + "=" * 60)
    response = input("Would you like to try interactive sentiment analysis? (y/n): ")
    if response.lower().startswith('y'):
        interactive_demo()
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
