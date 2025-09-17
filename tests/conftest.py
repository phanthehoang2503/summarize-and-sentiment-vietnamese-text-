"""
Test configuration and fixtures
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture
def sample_vietnamese_text():
    """Sample Vietnamese text for testing"""
    return "Tôi rất thích sản phẩm này. Nó hoạt động tốt và có thiết kế đẹp."

@pytest.fixture  
def sample_long_vietnamese_text():
    """Longer Vietnamese text for testing summarization"""
    return (
        "Tôi rất thích sản phẩm này. Nó hoạt động tốt và có thiết kế đẹp. "
        "Chất lượng rất tuyệt vời và dịch vụ khách hàng rất chu đáo. "
        "Tôi sẽ giới thiệu sản phẩm này cho bạn bè và gia đình. "
        "Giá cả cũng rất hợp lý so với chất lượng mà sản phẩm mang lại."
    )