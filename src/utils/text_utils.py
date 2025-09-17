"""
Text processing utilities for Vietnamese text
Simple and robust implementations for demo use
"""
import re
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

def clean_vietnamese_text(text: str) -> str:
    """Clean and normalize Vietnamese text with error handling"""
    if not isinstance(text, str):
        logger.warning("Input is not a string, returning empty string")
        return ""
    
    if not text.strip():
        return ""
    
    try:
        # Simple, reliable cleaning steps
        text = text.strip()
        
        # Normalize whitespace (convert all whitespace to single spaces)
        text = re.sub(r'\s+', ' ', text)
        
        # Fix punctuation spacing (ensure single space after sentence punctuation)
        text = re.sub(r'([.!?])([^\s])', r'\1 \2', text)
        
        # Remove excessive punctuation (limit to 3 consecutive chars)
        text = re.sub(r'([.!?]){4,}', r'\1\1\1', text)
        
        # Final cleanup
        text = text.strip()
        
        return text
        
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return text.strip()  # Return original stripped text as fallback

def validate_vietnamese_text(text: str) -> bool:
    """Simple validation for Vietnamese text content"""
    if not isinstance(text, str) or not text.strip():
        return False
    
    try:
        # Check for reasonable length
        if len(text) < 3 or len(text) > 50000:
            return False
        
        # Check if text contains at least some valid characters
        # Allow Vietnamese characters, Latin alphabet, numbers, and common punctuation
        valid_chars = re.findall(r'[a-zA-ZàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ0-9\s\.,!?\-()]', text)
        
        # Text should be at least 70% valid characters
        validity_ratio = len(''.join(valid_chars)) / len(text)
        return validity_ratio >= 0.7
        
    except Exception as e:
        logger.error(f"Error validating text: {e}")
        return False

def extract_sentences(text: str, max_sentences: Optional[int] = None) -> List[str]:
    """Extract sentences from Vietnamese text"""
    if not validate_vietnamese_text(text):
        return []
    
    try:
        # Simple sentence splitting on common Vietnamese sentence endings
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Filter out very short "sentences" (likely fragments)
        sentences = [s for s in sentences if len(s) > 10]
        
        # Limit number of sentences if requested
        if max_sentences:
            sentences = sentences[:max_sentences]
            
        return sentences
        
    except Exception as e:
        logger.error(f"Error extracting sentences: {e}")
        return [text]  # Return original text as single sentence fallback