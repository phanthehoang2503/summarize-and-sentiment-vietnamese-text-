# scripts/demo.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class VietnameseSummarizer:
    def __init__(self, model_path="../models/summarizer/checkpoint-1854"):
        """Initialize the summarizer"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
    
    def summarize(self, text, max_length=128, num_beams=4):
        """Generate summary for input text"""
        if not text.strip():
            return "Please provide some text to summarize."
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate summary
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
        
        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()

def interactive_demo():
    """Run interactive demo"""
    print("Vietnamese Text Summarization Demo")
    print("=" * 50)
    
    # Initialize summarizer
    summarizer = VietnameseSummarizer()
    
    print("\nInstructions:")
    print("- Enter Vietnamese text to summarize")
    print("- Type 'quit' to exit")
    print("- Type 'example' for a sample text")
    print("-" * 50)
    
    sample_text = """
    Sở Y tế TPHCM vừa công bố kết quả kiểm tra 16 cơ sở khám chữa bệnh tư nhân trên địa bàn thành phố. 
    Trong đó, nhiều phòng khám bị phạt hành chính do vi phạm về giá dịch vụ y tế và hoạt động khám chữa bệnh không phép. 
    Đặc biệt, Công ty TNHH Phòng khám Đa khoa Lians MMC tại quận 7 bị phạt 59,4 triệu đồng do nhiều vi phạm nghiêm trọng. 
    Các vi phạm bao gồm thu phí cao hơn quy định, sử dụng thiết bị chưa được phép và hoạt động ngoài phạm vi được cấp phép.
    """
    
    while True:
        print("\n" + "="*50)
        user_input = input("Enter text to summarize (or 'quit'/'example'): ").strip()
        
        if user_input.lower() == 'quit':
            print("Thank you for using the summarizer!")
            break
        elif user_input.lower() == 'example':
            user_input = sample_text
            print(f"Using example text:\n{user_input[:200]}...")
        
        if user_input:
            print("\nGenerating summary...")
            try:
                summary = summarizer.summarize(user_input)
                
                print(f"\nOriginal text length: {len(user_input.split())} words")
                print(f"Summary length: {len(summary.split())} words")
                print(f"Compression ratio: {len(summary.split())/len(user_input.split()):.2%}")
                print(f"\nSummary:\n{summary}")
                
            except Exception as e:
                print(f"Error generating summary: {e}")
        else:
            print("Please provide some text to summarize.")

if __name__ == "__main__":
    interactive_demo()
