"""
Unified Configuration Management
Single source of truth for all application settings
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


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


@dataclass
class Config:
    """Main configuration class"""
    project_root: Path = field(init=False)
    params: Dict[str, Any] = field(default_factory=dict)
    model_paths: ModelPaths = field(init=False)
    app: AppConfig = field(default_factory=AppConfig)
    
    def __post_init__(self):
        self.project_root = self._find_project_root()
        self.params = self._load_yaml_config()
        self.model_paths = self._setup_model_paths()
        
        # Override app config from environment or params
        app_config = self.params.get('app', {})
        self.app.host = os.getenv('HOST', app_config.get('host', self.app.host))
        self.app.port = int(os.getenv('PORT', app_config.get('port', self.app.port)))
        self.app.debug = os.getenv('DEBUG', '').lower() in ('true', '1')
        self.app.secret_key = os.getenv('SECRET_KEY', self.app.secret_key)
    
    def _find_project_root(self) -> Path:
        """Find project root by searching for params.yaml"""
        # Start from config file location
        current_path = Path(__file__).parent.parent.parent  # Go up to project root
        
        # Validate we found the right directory
        if (current_path / "params.yaml").exists():
            return current_path
            
        # Fallback: search upward from current working directory
        current_path = Path.cwd()
        for path in [current_path] + list(current_path.parents):
            if (path / "params.yaml").exists():
                return path
                
        raise FileNotFoundError(
            "Could not find params.yaml. Ensure you're running from the project directory."
        )
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from params.yaml with validation"""
        config_path = self.project_root / "params.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                
            # Validate essential configuration
            self._validate_config(config)
            return config
            
        except Exception as e:
            logging.warning(f"Could not load {config_path}: {e}. Using defaults.")
            return {}
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters"""
        # Check demo settings
        if 'demo' in config:
            demo = config['demo']
            max_len = demo.get('max_text_length', 10000)
            min_len = demo.get('min_text_length', 10)
            
            if max_len <= min_len:
                raise ValueError(f"max_text_length ({max_len}) must be greater than min_text_length ({min_len})")
            
            if max_len > 50000:  # Reasonable upper bound
                logging.warning(f"max_text_length ({max_len}) is very large, consider reducing for demo")
        
        # Validate quality modes
        if 'demo' in config and 'quality_modes' in config['demo']:
            modes = config['demo']['quality_modes']
            for mode_name, mode_config in modes.items():
                compression = mode_config.get('target_compression', 0.5)
                if not 0.1 <= compression <= 0.9:
                    raise ValueError(f"target_compression for {mode_name} must be between 0.1 and 0.9")
    
    def _setup_model_paths(self) -> ModelPaths:
        """Setup model-related paths"""
        models_dir = self.project_root / "models"
        cache_dir = self.project_root / "cache"
        
        return ModelPaths(
            sentiment_model_dir=models_dir / "sentiment" / "checkpoint-1182",
            summarizer_model_dir=models_dir / "summarizer", 
            cache_dir=cache_dir
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
    
    # Backward compatibility properties
    @property
    def sentiment_model_dir(self) -> Path:
        return self.model_paths.sentiment_model_dir
    
    @property
    def sentiment_cache_dir(self) -> Path:
        return self.model_paths.cache_dir
    
    @property
    def summarization_model_dir(self) -> Path:
        return self.model_paths.summarizer_model_dir
    
    @property
    def summarization_cache_dir(self) -> Path:
        return self.model_paths.cache_dir
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for specific model type"""
        return self.params.get('models', {}).get(model_type, {})
    
    def get_training_config(self, model_type: str) -> Dict[str, Any]:
        """Get training configuration for specific model type"""
        return self.params.get('training', {}).get(model_type, {})
    
    def __getattr__(self, name: str) -> Any:
        """Fallback to params for any missing attributes"""
        return self.params.get(name)


# Global config instance
_config: Optional[Config] = None

def get_config() -> Config:
    """Get the global configuration instance (lazy initialization)"""
    global _config
    if _config is None:
        _config = Config()
    return _config


# Backward compatibility
config = get_config()
settings = config