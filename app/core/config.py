"""
Unified Configuration Management
Single source of truth for all application settings
"""
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class ModelPaths:
    """Model-related paths configuration"""
    sentiment_model_dir: Path
    summarizer_model_dir: Path
    cache_dir: Path


@dataclass 
class AppConfig:
    """Application configuration"""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    secret_key: str = "vietnamese-text-analysis-academic-demo"


class Settings:
    """Centralized configuration manager with singleton pattern"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.project_root = self._find_project_root()
        self.params = self._load_yaml_config()
        self.model_paths = self._setup_model_paths()
        self.app = AppConfig()
        self._setup_logging()
        self._initialized = True
    
    def _find_project_root(self) -> Path:
        """Find project root by searching for params.yaml"""
        current_path = Path(__file__).parent
        for path in [current_path] + list(current_path.parents):
            if (path / "params.yaml").exists():
                return path
        raise FileNotFoundError("Could not find params.yaml - ensure you're in the project directory")
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from params.yaml"""
        config_path = self.project_root / "params.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logging.warning(f"Could not load {config_path}: {e}. Using defaults.")
            return {}
    
    def _setup_model_paths(self) -> ModelPaths:
        """Setup model-related paths"""
        models_dir = self.project_root / "models"
        cache_dir = self.project_root / "cache"
        
        return ModelPaths(
            sentiment_model_dir=models_dir / "sentiment" / "checkpoint-1182",
            summarizer_model_dir=models_dir / "summarizer", 
            cache_dir=cache_dir
        )
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.params.get('logging', {}).get('level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @property
    def data_dir(self) -> Path:
        """Data directory path"""
        return self.project_root / "data"
    
    @property
    def logs_dir(self) -> Path:
        """Logs directory path"""
        return self.project_root / "logs"
    
    @property
    def static_dir(self) -> Path:
        """Static files directory path"""
        return self.project_root / "static"
    
    @property
    def templates_dir(self) -> Path:
        """Templates directory path"""
        return self.project_root / "templates"
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for specific model type"""
        return self.params.get('models', {}).get(model_type, {})
    
    def get_training_config(self, model_type: str) -> Dict[str, Any]:
        """Get training configuration for specific model type"""
        return self.params.get('training', {}).get(model_type, {})


# Global settings instance
settings = Settings()

# Backward compatibility - matches the old config interface
class Config:
    """Backward compatibility wrapper"""
    def __init__(self):
        self._settings = settings
    
    @property
    def project_root(self) -> Path:
        return self._settings.project_root
    
    @property
    def data_dir(self) -> Path:
        return self._settings.data_dir
    
    @property
    def sentiment_model_dir(self) -> Path:
        return self._settings.model_paths.sentiment_model_dir
    
    @property
    def sentiment_cache_dir(self) -> Path:
        return self._settings.model_paths.cache_dir
    
    @property
    def summarization_model_dir(self) -> Path:
        return self._settings.model_paths.summarizer_model_dir
    
    @property
    def summarization_cache_dir(self) -> Path:
        return self._settings.model_paths.cache_dir
    
    def __getattr__(self, name: str) -> Any:
        """Fallback to params for any missing attributes"""
        return self._settings.params.get(name)


# Global config instance for backward compatibility
config = Config()

def get_config() -> Config:
    """Get the global configuration instance"""
    return config