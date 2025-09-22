"""
Vietnamese Text Analysis Web Application
Clean Flask application with modular structure
"""
import sys
import logging
from pathlib import Path
from flask import Flask, render_template, request

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_config

def setup_logging():
    """Configure simple logging for the application"""
    config = get_config()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(config.logs_dir / 'app.log')  # File output
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('datasets').setLevel(logging.ERROR)
    logging.getLogger('numexpr').setLevel(logging.ERROR)


def create_app(config_override=None):
    """Application factory pattern with proper logging setup"""
    # Setup logging first
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        config = get_config()
        logger.info("Starting Flask application creation")
        
        app = Flask(__name__,
                    template_folder=str(config.templates_dir),
                    static_folder=str(config.static_dir))
        
        # Configure app
        app.config.update({
            'SECRET_KEY': config.app.secret_key,
            'JSON_AS_ASCII': False,  # Support Vietnamese characters
            'JSONIFY_PRETTYPRINT_REGULAR': True,
            'MAX_CONTENT_LENGTH': 16 * 1024 * 1024  # 16MB limit
        })
        
        # Apply any configuration overrides
        if config_override:
            app.config.update(config_override)
            logger.info("Configuration overrides applied")
        
        # Register blueprints
        from app.api.analysis import analysis_bp
        from app.api.utils import utils_bp
        
        app.register_blueprint(analysis_bp)
        app.register_blueprint(utils_bp)
        logger.info("API blueprints registered successfully")
        
        # Main web routes
        @app.route('/')
        def index():
            """Main page"""
            logger.info("Index page requested")
            return render_template('index.html')
        
        # Error handlers with logging
        @app.errorhandler(404)
        def not_found_error(error):
            logger.warning(f"404 error: {request.url}")
            return {'error': 'Resource not found'}, 404
        
        @app.errorhandler(500)
        def internal_error(error):
            logger.error(f"500 error: {error}", exc_info=True)
            return {'error': 'Internal server error'}, 500
        
        @app.errorhandler(413)
        def file_too_large(error):
            logger.warning(f"File too large from {request.remote_addr}")
            return {'error': 'File too large'}, 413
        
        logger.info("Flask application created successfully")
        return app
        
    except Exception as e:
        print(f"CRITICAL: Failed to create Flask app: {e}")
        raise


def main():
    """Main entry point for the web application"""
    config = get_config()
    
    print("starting Vietnamese Text Analysis Academic Demo...")
    print(f"open your browser and go to: http://{config.app.host}:{config.app.port}")
    
    app = create_app()
    
    try:
        app.run(
            host=config.app.host,
            port=config.app.port,
            debug=config.app.debug,
            use_reloader=False  # Prevent reloader issues with models
        )
    except KeyboardInterrupt:
        print("\nshutting down gracefully...")
    except Exception as e:
        print(f"application failed to start: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()