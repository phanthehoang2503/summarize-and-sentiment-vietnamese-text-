"""
Unit tests for configuration management
"""

import pytest
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from app.core.config import get_config


def test_get_config():
    """Test basic configuration loading"""
    config = get_config()
    assert config is not None
    # Config is an object, not a dict
    assert hasattr(config, 'project_root')
    assert hasattr(config, 'params')


def test_config_has_required_keys():
    """Test that config has basic required attributes"""
    config = get_config()
    # Test that the config has the expected attributes
    assert hasattr(config, 'project_root')
    assert hasattr(config, 'params') 
    assert hasattr(config, 'app')
    assert config.params is not None