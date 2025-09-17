"""
Vietnamese Text Analysis Web Application
Clean Flask application with modular structure
"""
import sys
from pathlib import Path
from flask import Flask, render_template, send_from_directory
import logging

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.api.analysis import analysis_bp
from app.api.utils import utils_bp


def create_app(config_override=None):
    """
    Application factory pattern for creating Flask app
    
    Args:
        config_override: Optional configuration overrides
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__,
                template_folder=str(settings.templates_dir),
                static_folder=str(settings.static_dir))
    
    # Configure app
    app.config['SECRET_KEY'] = settings.app.secret_key
    app.config['JSON_AS_ASCII'] = False  # Support Vietnamese characters
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    
    # Apply any configuration overrides
    if config_override:
        app.config.update(config_override)
    
    # Register blueprints
    app.register_blueprint(analysis_bp)
    app.register_blueprint(utils_bp)
    
    # Setup logging
    if not app.debug:
        logging.basicConfig(level=logging.INFO)
    
    # Main web routes
    @app.route('/')
    def index():
        """Main page"""
        return render_template('index.html')
    
    @app.route('/favicon.ico')
    def favicon():
        """Serve favicon"""
        return send_from_directory(app.static_folder, 'favicon.ico', mimetype='image/vnd.microsoft.icon')
    
    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return {'error': 'Not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {'error': 'Internal server error'}, 500
    
    return app


def main():
    """Main entry point for the web application"""
    print("🎓 Starting Vietnamese Text Analysis Academic Demo...")
    print(f"📱 Open your browser and go to: http://{settings.app.host}:{settings.app.port}")
    
    app = create_app()
    
    try:
        app.run(
            host=settings.app.host,
            port=settings.app.port,
            debug=settings.app.debug,
            use_reloader=False  # Prevent reloader issues with models
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Application failed to start: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()