"""
Unit tests for configuration management
"""
import pytest
from pathlib import Path

from app.core.config import Settings, Config


class TestSettings:
    """Test suite for Settings class"""
    
    def test_singleton_pattern(self):
        """Test that Settings follows singleton pattern"""
        settings1 = Settings()
        settings2 = Settings()
        assert settings1 is settings2
    
    def test_project_root_exists(self):
        """Test that project root is found and exists"""
        settings = Settings()
        assert settings.project_root.exists()
        assert (settings.project_root / "params.yaml").exists()
    
    def test_model_paths_setup(self):
        """Test that model paths are properly configured"""
        settings = Settings()
        assert settings.model_paths.sentiment_model_dir.exists()
        assert settings.model_paths.summarizer_model_dir.exists()
        assert settings.model_paths.cache_dir.exists()
    
    def test_directory_properties(self):
        """Test directory property methods"""
        settings = Settings()
        
        # Test that paths exist
        assert settings.data_dir.exists()
        assert settings.static_dir.exists()
        assert settings.templates_dir.exists()


class TestConfig:
    """Test suite for Config backward compatibility wrapper"""
    
    def test_backward_compatibility(self):
        """Test that Config provides backward compatibility"""
        config = Config()
        
        # Test that all expected attributes exist
        assert hasattr(config, 'project_root')
        assert hasattr(config, 'sentiment_model_dir')
        assert hasattr(config, 'summarization_model_dir')
        assert hasattr(config, 'summarization_cache_dir')
        
        # Test that paths are Path objects
        assert isinstance(config.project_root, Path)
        assert isinstance(config.sentiment_model_dir, Path)
        assert isinstance(config.summarization_model_dir, Path)