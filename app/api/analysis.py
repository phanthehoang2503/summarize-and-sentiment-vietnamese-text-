"""
Text Analysis API Routes
Clean separation of API logic from business logic
"""
from flask import Blueprint, request, jsonify
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
from pathlib import Path
import logging
import uuid

from app.core.services import text_analysis_service

# Create blueprint for analysis APIs
analysis_bp = Blueprint('analysis', __name__, url_prefix='/api')
logger = logging.getLogger(__name__)

# Constants
MAX_TEXT_LENGTH = 50000  # Maximum characters for text input
ALLOWED_QUALITY_MODES = {'balanced', 'detailed'}
ALLOWED_FILE_EXTENSIONS = {'.txt', '.md', '.doc', '.docx'}

def generate_request_id():
    """Generate unique request ID for tracking"""
    return str(uuid.uuid4())[:8]

def validate_text_input(text: str, request_id: str) -> tuple[bool, str]:
    """Validate text input with detailed error messages"""
    if not text or not text.strip():
        return False, "Text content is required and cannot be empty"
    
    if len(text) > MAX_TEXT_LENGTH:
        logger.warning(f"Request {request_id}: Text too long ({len(text)} chars)")
        return False, f"Text too long. Maximum {MAX_TEXT_LENGTH} characters allowed"
    
    return True, ""

def validate_quality_mode(quality_mode: str) -> tuple[bool, str]:
    """Validate quality mode parameter"""
    if quality_mode not in ALLOWED_QUALITY_MODES:
        return False, f"Invalid quality_mode. Must be one of: {', '.join(ALLOWED_QUALITY_MODES)}"
    return True, ""


@analysis_bp.route('/summarize', methods=['POST'])
def api_summarize():
    """Text summarization API endpoint"""
    request_id = generate_request_id()
    logger.info(f"Request {request_id}: Summarization request received")
    
    try:
        if not request.is_json:
            raise BadRequest("Content-Type must be application/json")
        
        data = request.get_json()
        text = data.get('text', '').strip()
        quality_mode = data.get('quality_mode', 'balanced')
        
        # Validate inputs
        is_valid, error_msg = validate_text_input(text, request_id)
        if not is_valid:
            return jsonify({
                'success': False, 
                'error': error_msg,
                'request_id': request_id
            }), 400
        
        is_valid, error_msg = validate_quality_mode(quality_mode)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg, 
                'request_id': request_id
            }), 400
        
        logger.info(f"Request {request_id}: Processing text ({len(text)} chars, {quality_mode} mode)")
        result = text_analysis_service.summarize_text(text, quality_mode=quality_mode)
        
        result['request_id'] = request_id
        if result['success']:
            logger.info(f"Request {request_id}: Summarization completed successfully")
            return jsonify(result)
        else:
            logger.error(f"Request {request_id}: Summarization failed - {result.get('error', 'Unknown error')}")
            return jsonify(result), 500
            
    except BadRequest as e:
        logger.warning(f"Request {request_id}: Bad request - {e}")
        return jsonify({
            'success': False, 
            'error': str(e),
            'request_id': request_id
        }), 400
    except Exception as e:
        logger.error(f"Request {request_id}: Summarization API error - {e}", exc_info=True)
        return jsonify({
            'success': False, 
            'error': 'Internal server error - please try again later',
            'request_id': request_id
        }), 500


@analysis_bp.route('/sentiment', methods=['POST'])  
def api_sentiment():
    """Sentiment analysis API endpoint"""
    request_id = generate_request_id()
    logger.info(f"Request {request_id}: Sentiment analysis request received")
    
    try:
        if not request.is_json:
            raise BadRequest("Content-Type must be application/json")
        
        data = request.get_json()
        text = data.get('text', '').strip()
        
        # Validate inputs
        is_valid, error_msg = validate_text_input(text, request_id)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg,
                'request_id': request_id
            }), 400
        
        logger.info(f"Request {request_id}: Processing sentiment analysis ({len(text)} chars)")
        result = text_analysis_service.analyze_sentiment(text)
        
        result['request_id'] = request_id
        if result['success']:
            logger.info(f"Request {request_id}: Sentiment analysis completed successfully")
            return jsonify(result)
        else:
            logger.error(f"Request {request_id}: Sentiment analysis failed - {result.get('error', 'Unknown error')}")
            return jsonify(result), 500
            
    except BadRequest as e:
        logger.warning(f"Request {request_id}: Bad request - {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'request_id': request_id
        }), 400
    except Exception as e:
        logger.error(f"Request {request_id}: Sentiment API error - {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error - please try again later',
            'request_id': request_id
        }), 500


@analysis_bp.route('/analyze', methods=['POST'])
def api_analyze():
    """Combined analysis API endpoint"""
    request_id = generate_request_id()
    logger.info(f"Request {request_id}: Combined analysis request received")
    
    try:
        if not request.is_json:
            raise BadRequest("Content-Type must be application/json")
        
        data = request.get_json()
        text = data.get('text', '').strip()
        quality_mode = data.get('quality_mode', 'balanced')
        
        # Validate inputs
        is_valid, error_msg = validate_text_input(text, request_id)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg,
                'request_id': request_id
            }), 400
        
        is_valid, error_msg = validate_quality_mode(quality_mode)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg,
                'request_id': request_id
            }), 400
        
        logger.info(f"Request {request_id}: Processing combined analysis ({len(text)} chars, {quality_mode} mode)")
        result = text_analysis_service.analyze_combined(text, quality_mode=quality_mode)
        
        result['request_id'] = request_id
        if result['success']:
            logger.info(f"Request {request_id}: Combined analysis completed successfully")
            return jsonify(result)
        else:
            logger.error(f"Request {request_id}: Combined analysis failed - {result.get('error', 'Unknown error')}")
            return jsonify(result), 500
            
    except BadRequest as e:
        logger.warning(f"Request {request_id}: Bad request - {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'request_id': request_id
        }), 400
    except Exception as e:
        logger.error(f"Request {request_id}: Combined analysis API error - {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error - please try again later',
            'request_id': request_id
        }), 500


@analysis_bp.route('/upload', methods=['POST'])
def api_upload():
    """File upload API endpoint"""
    request_id = generate_request_id()
    logger.info(f"Request {request_id}: File upload request received")
    
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded',
                'request_id': request_id
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected',
                'request_id': request_id
            }), 400
        
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_FILE_EXTENSIONS:
            return jsonify({
                'success': False,
                'error': f'Unsupported file type. Allowed: {", ".join(ALLOWED_FILE_EXTENSIONS)}',
                'request_id': request_id
            }), 400
        
        # Read file content with encoding detection
        try:
            content = file.read().decode('utf-8')
        except UnicodeDecodeError:
            try:
                file.seek(0)  # Reset file pointer
                content = file.read().decode('utf-8-sig')  # Handle BOM
            except UnicodeDecodeError:
                logger.warning(f"Request {request_id}: File encoding error for {file.filename}")
                return jsonify({
                    'success': False, 
                    'error': 'Unable to decode file. Please ensure it\'s a UTF-8 text file.',
                    'request_id': request_id
                }), 400
        
        # Validate content length
        is_valid, error_msg = validate_text_input(content, request_id)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg,
                'request_id': request_id
            }), 400
        
        logger.info(f"Request {request_id}: Processing uploaded file {file.filename} ({len(content)} chars)")
        result = text_analysis_service.process_file_upload(content, file.filename)
        
        result['request_id'] = request_id
        if result['success']:
            logger.info(f"Request {request_id}: File upload processed successfully")
            return jsonify(result)
        else:
            logger.error(f"Request {request_id}: File upload processing failed - {result.get('error', 'Unknown error')}")
            return jsonify(result), 400
            
    except RequestEntityTooLarge:
        logger.warning(f"Request {request_id}: File too large")
        return jsonify({
            'success': False,
            'error': 'File too large. Maximum size allowed is 16MB',
            'request_id': request_id
        }), 413
    except Exception as e:
        logger.error(f"Request {request_id}: Upload API error - {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error - please try again later',
            'request_id': request_id
        }), 500