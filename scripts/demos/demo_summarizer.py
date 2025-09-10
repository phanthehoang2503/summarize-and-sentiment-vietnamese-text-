"""
Demo script showcasing the VietnameseSummarizer class
Run this to test your model with custom text or sample data
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from models import VietnameseSummarizer, create_summarizer, quick_summarize
    print("Successfully imported VietnameseSummarizer!")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def demo_quick_usage():
    """Demo the quick_summarize function"""
    print("\n" + "="*60)
    print("DEMO 1: Quick Summarization")
    print("="*60)
    
    sample_text = """
    Theo báo cáo của Bộ Y tế, tình hình dịch COVID-19 tại Việt Nam đang được kiểm soát tốt. 
    Trong 24 giờ qua, cả nước ghi nhận 145 ca mắc mới, giảm 20 ca so với hôm qua. 
    Các ca mắc chủ yếu được phát hiện trong khu cách ly và các ổ dịch đã được phong tỏa. 
    Bộ Y tế khuyến cáo người dân tiếp tục thực hiện nghiêm các biện pháp phòng chống dịch 
    như đeo khẩu trang, rửa tay thường xuyên và giữ khoảng cách an toàn.
    """
    
    print("Original text:")
    print(sample_text.strip())
    
    print("\nGenerating summary...")
    summary = quick_summarize(sample_text)
    
    print(f"\nGenerated summary:")
    print(f"'{summary}'")
    print(f"Length: {len(summary.split())} words")


def demo_class_usage():
    """Demo the full VietnameseSummarizer class"""
    print("\n" + "="*60)
    print("DEMO 2: Full Class Usage")
    print("="*60)
    
    print("Creating summarizer...")
    summarizer = create_summarizer(checkpoint="1854")
    
    info = summarizer.get_model_info()
    print(f"\nModel Info:")
    print(f"- Device: {info['device']}")
    print(f"- Model type: {info['model_type']}")
    print(f"- Vocab size: {info['vocab_size']}")
    
    # Test multiple texts
    test_texts = [
        """
        Giá vàng thế giới hôm nay tăng mạnh do lo ngại về lạm phát tại Mỹ. 
        Giá vàng giao ngay tăng 15 USD/ounce, lên mức 1.950 USD/ounce. 
        Các chuyên gia dự báo giá vàng có thể tiếp tục tăng trong thời gian tới 
        do USD suy yếu và các yếu tố địa chính trị bất ổn.
        """,
        """
        Đội tuyển bóng đá Việt Nam đã có chiến thắng ấn tượng 2-0 trước Indonesia 
        trong trận đấu vòng loại World Cup 2026. Hai bàn thắng được ghi ở phút 23 
        và phút 67 bởi Nguyễn Quang Hải và Phan Văn Đức. Với chiến thắng này, 
        Việt Nam duy trì vị trí đầu bảng với 15 điểm sau 6 trận đấu.
        """
    ]
    
    print(f"\nTesting batch summarization with {len(test_texts)} texts...")
    summaries = summarizer.summarize_batch(test_texts, show_progress=True)
    
    for i, (text, summary) in enumerate(zip(test_texts, summaries), 1):
        print(f"\n--- Example {i} ---")
        print(f"Original ({len(text.split())} words): {text.strip()[:100]}...")
        print(f"Summary ({len(summary.split())} words): {summary}")


def demo_evaluation():
    """Demo model evaluation"""
    print("\n" + "="*60)
    print("DEMO 3: Model Evaluation")
    print("="*60)
    
    try:
        # Create summarizer
        summarizer = create_summarizer()
        
        # Run evaluation on small sample
        print("Running evaluation on 10 samples...")
        results, predictions, references = summarizer.evaluate_on_dataset(
            num_samples=10,
            text_column="Contents",
            summary_column="Summary"
        )
        
        print(f"\nEvaluation Results:")
        print(f"ROUGE-1: {results['rouge1']:.4f}")
        print(f"ROUGE-2: {results['rouge2']:.4f}")
        print(f"ROUGE-L: {results['rougeL']:.4f}")
        print(f"Avg prediction length: {results['avg_prediction_length']:.1f} words")
        print(f"Avg reference length: {results['avg_reference_length']:.1f} words")
        
        # Find best and worst examples
        best, worst = summarizer.find_best_worst_examples(predictions, references)
        
        print(f"\nBest Example (ROUGE-1: {best['rouge1']:.4f}):")
        print(f"Prediction: {best['prediction']}")
        print(f"Reference: {best['reference']}")
        
        print(f"\nWorst Example (ROUGE-1: {worst['rouge1']:.4f}):")
        print(f"Prediction: {worst['prediction']}")
        print(f"Reference: {worst['reference']}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("This might be due to missing dependencies or data files.")


def main():
    """Run all demos"""
    print("Vietnamese Text Summarization Model Demo")
    print("="*60)
    
    try:
        # Demo 1: Quick usage
        demo_quick_usage()
        
        # Demo 2: Class usage
        demo_class_usage()
        
        # Demo 3: Evaluation (optional)
        try:
            demo_evaluation()
        except Exception as e:
            print(f"\nSkipping evaluation demo due to: {e}")
        
        print("\n" + "="*60)
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
