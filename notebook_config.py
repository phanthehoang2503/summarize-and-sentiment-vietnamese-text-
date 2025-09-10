"""
Notebook Configuration Helper
Use this in notebooks to load YAML configuration
"""
import sys
from pathlib import Path

# Auto-detect project root and add to path
def setup_notebook_config():
    """Setup configuration for notebook environment"""
    
    # Find project root (look for config.yaml)
    current_path = Path.cwd()
    project_root = None
    
    # Check current directory and parents
    for path in [current_path] + list(current_path.parents):
        if (path / "config.yaml").exists():
            project_root = path
            break
    
    if project_root is None:
        raise FileNotFoundError("Could not find config.yaml in current directory or parent directories")
    
    # Add project root to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Import configuration
    from config import config
    
    print(f"Configuration loaded from: {config.config_path}")
    print(f"Project root: {config.project_root}")
    
    return config

# For easy import in notebooks
def get_config():
    """Get project configuration (call setup first)"""
    try:
        from config import config
        return config
    except ImportError:
        return setup_notebook_config()

# Example usage for notebooks
def notebook_example():
    """Example of how to use config in notebooks"""
    
    # Setup configuration
    config = setup_notebook_config()
    
    # Get paths
    print("\nData Paths:")
    print(f"Raw data: {config.raw_data_dir}")
    print(f"Processed data: {config.processed_data_dir}")
    print(f"Sentiment data: {config.sentiment_data}")
    print(f"Summarization data: {config.summarization_data}")
    
    # Get model configurations
    print("\nModel Settings:")
    phobert_config = config.get_model_config("phobert")
    print(f"PhoBERT model: {phobert_config['name']}")
    print(f"PhoBERT cache: {config.sentiment_cache_dir}")
    
    vit5_config = config.get_model_config("vit5") 
    print(f"ViT5 model: {vit5_config['name']}")
    print(f"ViT5 cache: {config.summarization_cache_dir}")
    
    # Get training configurations
    print("\nTraining Settings:")
    sentiment_training = config.get_training_config("sentiment")
    print(f"Sentiment batch size: {sentiment_training['batch_size']}")
    print(f"Sentiment learning rate: {sentiment_training['learning_rate']}")
    
    summarization_training = config.get_training_config("summarization")
    print(f"Summarization batch size: {summarization_training['batch_size']}")
    print(f"Summarization learning rate: {summarization_training['learning_rate']}")
    
    return config

if __name__ == "__main__":
    # Example usage
    config = notebook_example()
