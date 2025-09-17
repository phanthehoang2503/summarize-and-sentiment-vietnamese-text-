"""
Health check and utility API routes
"""
from flask import Blueprint, jsonify
import sys
from pathlib import Path

from app.core.config import settings

# Create blueprint for utility APIs
utils_bp = Blueprint('utils', __name__, url_prefix='/api')


@utils_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'python_version': sys.version,
        'project_root': str(settings.project_root)
    })


@utils_bp.route('/status', methods=['GET'])
def system_status():
    """System status endpoint with more detailed information"""
    try:
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
                'project_root': str(settings.project_root),
                'models_dir': str(settings.model_paths.sentiment_model_dir.parent),
                'cache_dir': str(settings.model_paths.cache_dir)
            }
        }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            'status': 'degraded',
            'error': str(e),
            'services': {
                'text_analysis': 'error',
                'summarization': 'error',
                'sentiment_analysis': 'error', 
                'combined_pipeline': 'error'
            }
        }), 500