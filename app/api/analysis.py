"""
Text Analysis API Routes
Clean separation of API logic from business logic
"""
from flask import Blueprint, request, jsonify
from werkzeug.exceptions import BadRequest
import logging

from app.core.services import text_analysis_service

# Create blueprint for analysis APIs
analysis_bp = Blueprint('analysis', __name__, url_prefix='/api')
logger = logging.getLogger(__name__)


@analysis_bp.route('/summarize', methods=['POST'])
def api_summarize():
    """Text summarization API endpoint"""
    try:
        if not request.is_json:
            raise BadRequest("Content-Type must be application/json")
        
        data = request.get_json()
        text = data.get('text', '').strip()
        quality_mode = data.get('quality_mode', 'balanced')  # Default to balanced
        
        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400
        
        result = text_analysis_service.summarize_text(text, quality_mode=quality_mode)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except BadRequest as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Summarization API error: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


@analysis_bp.route('/sentiment', methods=['POST'])  
def api_sentiment():
    """Sentiment analysis API endpoint"""
    try:
        if not request.is_json:
            raise BadRequest("Content-Type must be application/json")
        
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400
        
        result = text_analysis_service.analyze_sentiment(text)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except BadRequest as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Sentiment API error: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


@analysis_bp.route('/analyze', methods=['POST'])
def api_analyze():
    """Combined analysis API endpoint"""
    try:
        if not request.is_json:
            raise BadRequest("Content-Type must be application/json")
        
        data = request.get_json()
        text = data.get('text', '').strip()
        quality_mode = data.get('quality_mode', 'balanced')  # Default to balanced
        
        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400
        
        result = text_analysis_service.analyze_combined(text, quality_mode=quality_mode)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except BadRequest as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Combined analysis API error: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


@analysis_bp.route('/upload', methods=['POST'])
def api_upload():
    """File upload API endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Read file content with encoding detection
        try:
            content = file.read().decode('utf-8')
        except UnicodeDecodeError:
            try:
                content = file.read().decode('utf-8-sig')  # Handle BOM
            except UnicodeDecodeError:
                return jsonify({
                    'success': False, 
                    'error': 'Unable to decode file. Please ensure it\'s a UTF-8 text file.'
                }), 400
        
        result = text_analysis_service.process_file_upload(content, file.filename)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Upload API error: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500