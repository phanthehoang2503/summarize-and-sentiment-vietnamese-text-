#!/usr/bin/env python3
"""
Simple Demo Script for Vietnamese Text Analysis
Demonstrates both sentiment analysis and text summarization
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def demo_sentiment():
    """Demo sentiment analysis"""
    print("🔍 Sentiment Analysis Demo")
    print("-" * 30)
    
    sample_texts = [
        "Tôi rất thích sản phẩm này!",
        "Dịch vụ tệ quá, không hài lòng.",
        "Sản phẩm bình thường, không có gì đặc biệt."
    ]
    
    try:
        from src.models.sentiment import VietnameseSentimentAnalyzer
        analyzer = VietnameseSentimentAnalyzer()
        
        for text in sample_texts:
            result = analyzer.predict(text)
            print(f"Text: {text}")
            print(f"Sentiment: {result}")
            print()
    except Exception as e:
        print(f"Error in sentiment demo: {e}")

def demo_summarization():
    """Demo text summarization"""
    print("📄 Text Summarization Demo")
    print("-" * 30)
    
    sample_text = """
    Trí tuệ nhân tạo (AI) đang phát triển rất nhanh trong những năm gần đây. 
    AI được ứng dụng trong nhiều lĩnh vực như y tế, giáo dục, giao thông và tài chính. 
    Việc phát triển AI mang lại nhiều lợi ích nhưng cũng đặt ra những thách thức về 
    đạo đức và việc làm. Cần có những quy định phù hợp để đảm bảo AI phát triển 
    một cách có trách nhiệm và bền vững.
    """
    
    try:
        from src.models.summarizer import VietnameseSummarizer
        summarizer = VietnameseSummarizer()
        
        print(f"Original text:\n{sample_text.strip()}")
        print()
        
        summary = summarizer.summarize(sample_text)
        print(f"Summary: {summary}")
        print()
    except Exception as e:
        print(f"Error in summarization demo: {e}")

def main():
    """Main demo function"""
    print("🇻🇳 Vietnamese Text Analysis Demo")
    print("=" * 50)
    print()
    
    try:
        demo_sentiment()
        print("\n" + "=" * 50 + "\n")
        demo_summarization()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Demo failed: {e}")
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    main()