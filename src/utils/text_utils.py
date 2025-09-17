import re

def clean_vietnamese_text(text):
        """Clean and normalize Vietnamese text.

        Steps:
            1. Normalize whitespace
            2. Collapse repeated punctuation (.,!,?) to a single final char
            3. Ensure single space after sentence punctuation
            4. Remove duplicated spaces again
        """
        if not isinstance(text, str):
                return ""

        txt = text.strip()
        # Normalize whitespace early
        txt = re.sub(r'\s+', ' ', txt)
        # Collapse runs of punctuation like '!!', '??', '..'
        txt = re.sub(r'([.!?])\1+', r'\1', txt)
        # Replace stray multi-dot ellipsis patterns (>1) with single period
        txt = re.sub(r'\.{2,}', '.', txt)
        # Space after punctuation if followed by word character
        txt = re.sub(r'([.!?])(\S)', r'\1 \2', txt)
        # Collapse any double spaces produced
        txt = re.sub(r'\s{2,}', ' ', txt)
        return txt.strip()

def validate_vietnamese_text(text):
    """Validate if text contains valid Vietnamese characters"""
    vietnamese_pattern = r'^[a-zA-Z0-9\s\.,!?\(\)\[\]àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]+$'
    return bool(re.match(vietnamese_pattern, text))