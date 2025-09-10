import re

def clean_vietnamese_text(text):
    """Clean and normalize Vietnamese text"""
    if not isinstance(text, str):
        return ""
        
    # Remove excessive spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common punctuation issues
    text = text.replace('..','.').replace('...','.').replace(',.', '.')
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.!?])', r'\1 ', text)
    
    return text.strip()

def validate_vietnamese_text(text):
    """Validate if text contains valid Vietnamese characters"""
    vietnamese_pattern = r'^[a-zA-Z0-9\s\.,!?\(\)\[\]àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]+$'
    return bool(re.match(vietnamese_pattern, text))