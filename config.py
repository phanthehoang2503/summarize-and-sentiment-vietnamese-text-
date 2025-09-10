"""
YAML-Based Project Configuration
Centralized configuration management using YAML files
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

class ProjectConfig:
    """Project configuration manager using YAML"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to config.yaml file. If None, auto-detect.
        """
        self.project_root = self._find_project_root()
        
        if config_path is None:
            config_path = self.project_root / "config.yaml"
        else:
            config_path = Path(config_path)
            
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_paths()
        self._setup_logging()
    
    def _find_project_root(self) -> Path:
        """Find project root directory"""
        current = Path(__file__).parent
        
        # Look for config.yaml, notebooks folder, or data folder
        while current != current.parent:
            if any((current / marker).exists() for marker in 
                   ["config.yaml", "notebooks", "data", ".git"]):
                return current
            current = current.parent
        
        return Path(__file__).parent
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing YAML config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if YAML file is not available"""
        return {
            "paths": {
                "data": {"processed": "data/processed"},
                "models": {"base": "models"},
                "cache": {"base": "cache"}
            },
            "models": {
                "phobert": {"name": "vinai/phobert-base"},
                "vit5": {"name": "VietAI/vit5-base"}
            }
        }
    
    def _setup_paths(self):
        """Setup path attributes from configuration"""
        paths_config = self.config.get("paths", {})
        
        # Core directories
        self.data_dir = self.project_root / paths_config.get("data", {}).get("base", "data")
        self.raw_data_dir = self.project_root / paths_config.get("data", {}).get("raw", "data/raw")
        self.processed_data_dir = self.project_root / paths_config.get("data", {}).get("processed", "data/processed")
        self.cache_dir = self.project_root / paths_config.get("cache", {}).get("base", "cache")
        self.models_dir = self.project_root / paths_config.get("models", {}).get("base", "models")
        self.scripts_dir = self.project_root / paths_config.get("scripts", "scripts")
        self.notebooks_dir = self.project_root / paths_config.get("notebooks", "notebooks")
        
        # Data files
        data_files = self.config.get("data_files", {})
        self.sentiment_data = self.project_root / data_files.get("sentiment", {}).get("processed", "data/processed/reviews_clean.csv")
        self.summarization_data = self.project_root / data_files.get("summarization", {}).get("processed", "data/processed/articles_clean.csv")
        self.stopwords_file = self.project_root / paths_config.get("data", {}).get("stopwords", "data/vietnamese-stopwords.txt")
        
        # Model directories
        sentiment_config = self.config.get("models", {}).get("phobert", {})
        summarization_config = self.config.get("models", {}).get("vit5", {})
        
        self.sentiment_model_dir = self.project_root / sentiment_config.get("output_dir", "models/sentiment")
        self.sentiment_cache_dir = self.project_root / sentiment_config.get("cache_dir", "cache/vinai/phobert-base")
        
        self.summarization_model_dir = self.project_root / summarization_config.get("output_dir", "models/summarizer")
        self.summarization_cache_dir = self.project_root / summarization_config.get("cache_dir", "cache/VietAI/vit5-base")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get("logging", {})
        
        log_level = getattr(logging, log_config.get("level", "INFO"))
        log_format = log_config.get("format", "%(asctime)s - %(levelname)s - %(message)s")
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[logging.StreamHandler()] if log_config.get("console", True) else []
        )
        
        # File logging if specified
        if "file" in log_config:
            log_file = self.project_root / log_config["file"]
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(file_handler)
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        return self.config.get("models", {}).get(model_type, {})
    
    def get_training_config(self, task: str) -> Dict[str, Any]:
        """Get training configuration for a specific task"""
        return self.config.get("training", {}).get(task, {})
    
    def get_preprocessing_config(self, task: str = None) -> Dict[str, Any]:
        """Get preprocessing configuration"""
        preprocessing = self.config.get("preprocessing", {})
        if task:
            return preprocessing.get(task, {})
        return preprocessing
    
    def ensure_directories(self):
        """Create all necessary directories"""
        directories = [
            self.data_dir, self.raw_data_dir, self.processed_data_dir,
            self.cache_dir, self.models_dir, self.sentiment_model_dir,
            self.summarization_model_dir, self.sentiment_cache_dir,
            self.summarization_cache_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"All directories created/verified")
    
    def get_paths_info(self) -> Dict[str, Any]:
        """Get information about project paths"""
        return {
            "project_root": str(self.project_root),
            "config_file": str(self.config_path),
            "data_dir_exists": self.data_dir.exists(),
            "raw_data_exists": self.raw_data_dir.exists(),
            "processed_data_exists": self.processed_data_dir.exists(),
            "cache_exists": self.cache_dir.exists(),
            "models_exists": self.models_dir.exists(),
            "sentiment_data_exists": self.sentiment_data.exists(),
            "summarization_data_exists": self.summarization_data.exists(),
            "stopwords_exists": self.stopwords_file.exists()
        }
    
    def save_config(self, config_dict: Dict[str, Any] = None):
        """Save current configuration to YAML file"""
        config_to_save = config_dict or self.config
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Configuration saved to {self.config_path}")

config = ProjectConfig()

# Convenience exports for backward compatibility
PROJECT_ROOT = config.project_root
DATA_DIR = config.data_dir
RAW_DATA_DIR = config.raw_data_dir
PROCESSED_DATA_DIR = config.processed_data_dir
CACHE_DIR = config.cache_dir
MODELS_DIR = config.models_dir

SENTIMENT_PROCESSED = config.sentiment_data
SUMMARIZATION_PROCESSED = config.summarization_data
STOPWORDS_FILE = config.stopwords_file

# Model directories
SENTIMENT_MODEL_DIR = config.sentiment_model_dir
SENTIMENT_CACHE_DIR = config.sentiment_cache_dir
SUMMARIZATION_MODEL_DIR = config.summarization_model_dir
SUMMARIZATION_CACHE_DIR = config.summarization_cache_dir

SENTIMENT_CONFIG = config.get_training_config("sentiment")
SUMMARIZATION_CONFIG = config.get_training_config("summarization")

PHOBERT_CONFIG = config.get_model_config("phobert")
VIT5_CONFIG = config.get_model_config("vit5")

def ensure_directories():
    """Ensure all directories exist"""
    return config.ensure_directories()

def get_paths_info():
    """Get project paths information"""
    return config.get_paths_info()

if __name__ == "__main__":
    ensure_directories()
    info = get_paths_info()
    
    print("\nProject Configuration:")
    print(f"  Config file: {config.config_path}")
    print(f"  Project root: {PROJECT_ROOT}")
    
    print("\nPath Status:")
    for key, value in info.items():
        status = "OK" if value else "Missing"
        print(f"  {status} {key}: {value}")
    
    print("\nModel Configurations:")
    print(f"  PhoBERT: {PHOBERT_CONFIG.get('name')}")
    print(f"  ViT5: {VIT5_CONFIG.get('name')}")
    
    print("\nTraining Configurations:")
    print(f"  Sentiment batch size: {SENTIMENT_CONFIG.get('batch_size')}")
    print(f"  Summarization batch size: {SUMMARIZATION_CONFIG.get('batch_size')}")
