import cloudinary
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from .config import Config
from .routes import register_routes

# Initialize extensions (without app binding yet)
db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()


def create_app(config_object: type[Config] = Config) -> Flask:
    """Create and configure Flask application."""
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config.from_object(config_object)

    # Initialize database + migrate
    db.init_app(app)
    migrate.init_app(app, db)

    # Import models for migrations to detect them
    from app.models.analysis_entity import HealthAnalysis

    # Configure Cloudinary
    cloudinary.config(
        cloud_name=app.config.get("CLOUDINARY_CLOUD_NAME", ""),
        api_key=app.config.get("CLOUDINARY_API_KEY", ""),
        api_secret=app.config.get("CLOUDINARY_API_SECRET", "")
    )

    # Register routes
    register_routes(app)
    
    # Initialize JWT
    jwt.init_app(app)
    
    return app


# Create default app instance for WSGI servers
app = create_app()
