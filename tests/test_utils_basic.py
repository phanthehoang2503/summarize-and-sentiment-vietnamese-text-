import sys
from pathlib import Path

# Add project root src to path for direct invocation without pytest -m
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
import pytest
from src.utils.text_utils import clean_vietnamese_text, validate_vietnamese_text

def test_clean_vietnamese_text_basic():
    raw = "Xin   chao!!  Day la  test.."
    cleaned = clean_vietnamese_text(raw)
    assert '  ' not in cleaned  # no double spaces
    assert cleaned.count('..') == 0  # collapsed periods
    assert cleaned.endswith('.')  # final punctuation present

@pytest.mark.parametrize("text,expected", [
    ("Xin chào Việt Nam!", True),
    ("Hello123", True),
    ("@@@@@", False),
])
def test_validate_vietnamese_text(text, expected):
    assert validate_vietnamese_text(text) == expected


if __name__ == "__main__": 
    import sys as _sys
    # Run only this file's tests
    _sys.exit(pytest.main([__file__]))
