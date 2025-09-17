#!/usr/bin/env python3
"""
Demo script for Combined Summarization + Sentiment Analysis Pipeline
"""
import sys
from pathlib import Path

# Add project root to path for legacy compatibility
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pipeline import create_pipeline
from app.core.config import get_config

# Get configuration
config = get_config()


def demo_simple_pipeline():
    """Demo simple text processing with pipeline"""
    print("Simple Pipeline Demo")
    print("=" * 50)
    
    # Create pipeline
    print("Creating pipeline...")
    pipeline = create_pipeline()
    
    # Test texts
    test_texts = [
        """
        Hôm nay tôi đã có một trải nghiệm tuyệt vời tại nhà hàng mới ở trung tâm thành phố. 
        Món ăn rất ngon với nhiều món đặc sản địa phương được chế biến một cách tinh tế. 
        Dịch vụ của nhân viên rất chu đáo, họ luôn chăm sóc khách hàng một cách tận tình. 
        Không gian nhà hàng được thiết kế rất đẹp và sang trọng, tạo cảm giác thoải mái cho thực khách. 
        Giá cả cũng rất hợp lý so với chất lượng món ăn và dịch vụ mà họ cung cấp. 
        Tôi chắc chắn sẽ quay lại đây lần nữa và giới thiệu cho bạn bè cùng đến thưởng thức.
        """,
        """
        Tôi rất thất vọng với sản phẩm này. Chất lượng không như quảng cáo, rất tệ và không đáng tiền. 
        Dịch vụ khách hàng cũng không tốt, nhân viên không nhiệt tình giải đáp thắc mắc. 
        Thời gian giao hàng bị chậm trễ so với cam kết ban đầu. 
        Đóng gói sản phẩm cũng không cẩn thận, có nhiều vết xước và hư hỏng. 
        Tôi sẽ không mua sản phẩm của công ty này nữa và không khuyến khích ai khác mua.
        """,
        """
        Chiếc điện thoại này có nhiều tính năng thú vị và hiện đại. Camera chụp ảnh khá tốt trong điều kiện ánh sáng bình thường. 
        Pin có thể sử dụng cả ngày mà không cần sạc. Màn hình hiển thị rõ nét và màu sắc sống động. 
        Tuy nhiên, giá cả hơi cao so với các sản phẩm cùng phân khúc trên thị trường. 
        Thiết kế cũng không có gì đặc biệt so với các mẫu khác. Nhìn chung, đây là một sản phẩm ổn.
        """
    ]
    
    # Process each text
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Text {i} ---")
        print(f"Original: {text.strip()[:100]}...")
        
        # Analyze with pipeline
        result = pipeline.analyze(text, include_original_text=False)
        
        if result.get("success"):
            print(f"Analysis method: {result['analysis_method']}")
            if result['used_summarization']:
                print(f"Summary: {result['summary']}")
                print(f"Length: {result['original_length']} → {result['summary_length']} chars (ratio: {result['compression_ratio']:.2f})")
            else:
                print(f"Text analyzed directly (too short for summarization)")
                print(f"Length: {result['original_length']} characters")
            
            sentiment = result['sentiment']
            print(f"Sentiment: {sentiment['predicted_label'].upper()}")
            print(f"Confidence: {sentiment['confidence']:.3f}")
            
            if 'probabilities' in sentiment:
                print("Probabilities:")
                for label, prob in sentiment['probabilities'].items():
                    print(f"  {label}: {prob:.3f}")
        else:
            print(f"Error: {result.get('error')}")


def demo_batch_processing():
    """Demo batch processing with pipeline"""
    print("\n\nBatch Processing Demo")
    print("=" * 50)
    
    pipeline = create_pipeline()
    
    reviews = [
        "Sản phẩm tốt, chất lượng ổn, sẽ mua lại.",
        "Dịch vụ tệ, nhân viên không thân thiện, rất thất vọng.",
        "Món ăn ngon nhưng giá hơi đắt, không gian đẹp.",
        "Giao hàng nhanh, đóng gói cẩn thận, rất hài lòng.",
        "Chất lượng không như mong đợi, sản phẩm bị lỗi."
    ]
    
    print(f"Processing {len(reviews)} reviews...")
    
    results = pipeline.analyze_batch(
        reviews,
        max_summary_length=64,
        batch_size=2,
        show_progress=True
    )
    
    print("\nBatch Results:")
    for i, (review, result) in enumerate(zip(reviews, results), 1):
        if result.get("success"):
            sentiment = result['sentiment']['predicted_label']
            confidence = result['sentiment']['confidence']
            summary = result['summary']
            print(f"{i}. {sentiment.upper()} ({confidence:.3f}): {summary}")
        else:
            print(f"{i}. ERROR: {result.get('error')}")


def demo_long_document():
    """Demo processing long document by chunks"""
    print("\n\nLong Document Demo")
    print("=" * 50)
    
    # Create pipeline
    pipeline = create_pipeline()
    
    # Long document
    long_text = """
    Trong thời đại công nghệ 4.0, việc ứng dụng trí tuệ nhân tạo vào các lĩnh vực khác nhau đang trở thành xu hướng phát triển mạnh mẽ. 
    
    Ngành y tế là một trong những lĩnh vực được hưởng lợi nhiều nhất từ sự phát triển của AI. Các hệ thống chẩn đoán thông minh có thể phân tích hình ảnh y khoa với độ chính xác cao, giúp bác sĩ đưa ra quyết định điều trị nhanh chóng và chính xác hơn. Điều này không chỉ cải thiện chất lượng chăm sóc sức khỏe mà còn giảm thiểu chi phí và thời gian điều trị.
    
    Trong lĩnh vực giáo dục, AI đang tạo ra những thay đổi tích cực đáng kể. Các nền tảng học tập thông minh có thể cá nhân hóa quá trình học tập cho từng học sinh, phân tích điểm mạnh và điểm yếu để đưa ra lộ trình học tập phù hợp. Giáo viên cũng được hỗ trợ trong việc tạo ra nội dung giảng dạy và đánh giá kết quả học tập một cách hiệu quả hơn.
    
    Ngành giao thông vận tải cũng đang trải qua cuộc cách mạng với sự xuất hiện của xe tự lái và hệ thống quản lý giao thông thông minh. Điều này hứa hẹn sẽ giảm thiểu tai nạn giao thông, tối ưu hóa luồng di chuyển và giảm ô nhiễm môi trường.
    
    Tuy nhiên, bên cạnh những lợi ích to lớn, việc ứng dụng AI cũng đặt ra nhiều thách thức và lo ngại. Vấn đề về quyền riêng tư dữ liệu, an ninh mạng, và tác động đến việc làm của con người đang được quan tâm và thảo luận rộng rãi. 
    
    Để tận dụng tối đa lợi ích của AI và giảm thiểu rủi ro, cần có sự phối hợp chặt chẽ giữa các nhà nghiên cứu, doanh nghiệp, và cơ quan quản lý nhà nước. Việc xây dựng khung pháp lý phù hợp và đào tạo nguồn nhân lực chất lượng cao cũng là những yếu tố then chốt cho sự phát triển bền vững của AI trong tương lai.
    """
    
    print(f"Processing document ({len(long_text)} characters)...")
    
    # Process long document
    result = pipeline.analyze_document(
        long_text,
        chunk_size=800,
        overlap=100,
        combine_summaries=True,
        max_summary_length=100,
        include_original_text=False
    )
    
    if result.get("success"):
        print(f"Document processed successfully!")
        print(f"Original length: {result['original_length']} characters")
        print(f"Number of chunks: {result['num_chunks']}")
        print(f"Combined summary: {result['combined_summary']}")
        print(f"Summary length: {result['combined_summary_length']} characters")
        print(f"Compression ratio: {result['compression_ratio']:.2f}")
        
        sentiment = result['sentiment']
        print(f"Overall sentiment: {sentiment['predicted_label'].upper()}")
        print(f"Confidence: {sentiment['confidence']:.3f}")
        
        print("\nChunk details:")
        for i, chunk_result in enumerate(result['chunk_results'], 1):
            if chunk_result.get("success"):
                chunk_sentiment = chunk_result['sentiment']['predicted_label']
                chunk_conf = chunk_result['sentiment']['confidence']
                print(f"  Chunk {i}: {chunk_sentiment.upper()} ({chunk_conf:.3f})")
    else:
        print(f"Error: {result.get('error')}")


def interactive_pipeline():
    """Interactive pipeline demo"""
    print("\n\nInteractive Pipeline Demo")
    print("=" * 50)
    print("Enter Vietnamese text to summarize and analyze sentiment (type 'quit' to exit)")
    
    pipeline = create_pipeline()
    
    while True:
        try:
            user_input = input("\nEnter text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not user_input:
                print("Please enter some text.")
                continue
            
            print("Processing...")
            
            # Analyze with pipeline
            result = pipeline.analyze(
                user_input,
                max_summary_length=100,
                include_original_text=False
            )
            
            if result.get("success"):
                print(f"\nResults:")
                print(f"Analysis method: {result['analysis_method']}")
                if result['used_summarization']:
                    print(f"Summary: {result['summary']}")
                    print(f"Compression: {result['original_length']} → {result['summary_length']} chars ({result['compression_ratio']:.2f})")
                else:
                    print(f"Direct analysis (text length: {result['original_length']} chars)")
                
                sentiment = result['sentiment']
                print(f"Sentiment: {sentiment['predicted_label'].upper()}")
                print(f"Confidence: {sentiment['confidence']:.3f}")
                
                if 'probabilities' in sentiment:
                    print("Probabilities:")
                    for label, prob in sentiment['probabilities'].items():
                        bar_length = int(prob * 15)
                        bar = "█" * bar_length + "░" * (15 - bar_length)
                        print(f"  {label:8s}: {bar} {prob:.3f}")
            else:
                print(f"Error: {result.get('error')}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main demo function"""
    print("Vietnamese Summarization + Sentiment Analysis Pipeline Demo")
    print("=" * 70)
    
    # Run demos
    demo_simple_pipeline()
    demo_batch_processing()
    demo_long_document()
    
    # Ask if user wants interactive demo
    print("\n" + "=" * 70)
    response = input("Would you like to try interactive pipeline? (y/n): ")
    if response.lower().startswith('y'):
        interactive_pipeline()
    
    print("\nPipeline demo completed!")


if __name__ == "__main__":
    main()
