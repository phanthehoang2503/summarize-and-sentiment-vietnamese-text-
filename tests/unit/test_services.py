"""
Unit tests for text analysis services
"""
import pytest
from unittest.mock import Mock, patch

from app.core.services import TextAnalysisService


class TestTextAnalysisService:
    """Test suite for TextAnalysisService"""
    
    def setup_method(self):
        """Setup test environment"""
        self.service = TextAnalysisService()
    
    def test_is_vietnamese_text(self):
        """Test Vietnamese text detection"""
        vietnamese_text = "Tôi rất thích sản phẩm này"
        english_text = "I really like this product"
        
        assert self.service.is_vietnamese_text(vietnamese_text) is True
        assert self.service.is_vietnamese_text(english_text) is False
    
    def test_count_tokens(self):
        """Test token counting"""
        text = "Tôi rất thích sản phẩm này"
        assert self.service.count_tokens(text) == 6
    
    def test_summarize_text_empty_input(self):
        """Test summarization with empty input"""
        result = self.service.summarize_text("")
        assert result['success'] is False
        assert 'error' in result
    
    def test_analyze_sentiment_empty_input(self):
        """Test sentiment analysis with empty input"""
        result = self.service.analyze_sentiment("")
        assert result['success'] is False
        assert 'error' in result
    
    def test_analyze_combined_empty_input(self):
        """Test combined analysis with empty input"""
        result = self.service.analyze_combined("")
        assert result['success'] is False
        assert 'error' in result
    
    def test_process_file_upload_empty_content(self):
        """Test file upload with empty content"""
        result = self.service.process_file_upload("", "test.txt")
        assert result['success'] is False
        assert 'error' in result
    
    def test_process_file_upload_valid_content(self):
        """Test file upload with valid content"""
        content = "Tôi rất thích sản phẩm này"
        result = self.service.process_file_upload(content, "test.txt")
        assert result['success'] is True
        assert result['content'] == content
        assert result['filename'] == "test.txt"
        assert result['tokens'] == 6