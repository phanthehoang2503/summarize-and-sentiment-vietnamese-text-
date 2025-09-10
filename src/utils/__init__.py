# src/utils/__init__.py
from .data_analysis import (
    load_and_sample_data,
    calculate_text_metrics,
    analyze_compression_ratio,
    get_data_quality_report,
    print_data_summary
)

from .path_utils import (
    get_project_root,
    get_data_path,
    get_model_path,
    get_cache_path,
    ensure_path_exists
)

from .text_utils import (
    clean_vietnamese_text,
    validate_vietnamese_text
)

# Optional imports that may not be available in all environments
try:
    from .visualization import (
        plot_length_distributions,
        plot_compression_ratio,
        plot_sentiment_distribution,
        plot_cross_dataset_comparison
    )
except ImportError:
    # Define dummy functions so imports don't fail
    def plot_length_distributions(*args, **kwargs):
        pass
    
    def plot_compression_ratio(*args, **kwargs):
        pass
    
    def plot_sentiment_distribution(*args, **kwargs):
        pass
    
    def plot_cross_dataset_comparison(*args, **kwargs):
        pass
