"""
Health check and utility API routes
"""
from flask import Blueprint, jsonify
import sys
import logging
from pathlib import Path

from app.core.config import get_config

# Create blueprint for utility APIs
utils_bp = Blueprint('utils', __name__, url_prefix='/api')
logger = logging.getLogger(__name__)


@utils_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        config = get_config()
        return jsonify({
            'status': 'healthy',
            'version': '1.0.0',
            'python_version': sys.version,
            'project_root': str(config.project_root)
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': 'Configuration error'
        }), 500


@utils_bp.route('/status', methods=['GET'])
def system_status():
    """System status endpoint with more detailed information"""
    try:
        config = get_config()
        
        # Test model imports
        from app.core.services import text_analysis_service
        
        status = {
            'status': 'operational',
            'services': {
                'text_analysis': 'available',
                'summarization': 'available', 
                'sentiment_analysis': 'available',
                'combined_pipeline': 'available'
            },
            'configuration': {
                'project_root': str(config.project_root),
                'models_dir': str(config.model_paths.sentiment_model_dir.parent),
                'cache_dir': str(config.model_paths.cache_dir)
            }
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        return jsonify({
            'status': 'degraded',
            'error': 'Service initialization error',
            'services': {
                'text_analysis': 'error',
                'summarization': 'error',
                'sentiment_analysis': 'error', 
                'combined_pipeline': 'error'
            }
        }), 500


@utils_bp.route('/performance', methods=['GET'])
def performance_stats():
    """Get performance and caching statistics"""
    try:
        from app.core.services import text_analysis_service
        stats = text_analysis_service.get_performance_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Performance stats failed: {e}")
        return jsonify({
            'error': 'Failed to get performance stats',
            'details': str(e)
        }), 500


@utils_bp.route('/cache/clear', methods=['POST'])
def clear_caches():
    """Clear all caches (admin endpoint)"""
    try:
        from app.core.services import text_analysis_service
        result = text_analysis_service.clear_caches()
        logger.info("Caches cleared via API")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        return jsonify({
            'error': 'Failed to clear caches',
            'details': str(e)
        }), 500