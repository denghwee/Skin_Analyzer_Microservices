import cloudinary
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager

from .config import Config
from .routes import register_routes

db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()

def create_app(config_object=Config):
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Load config
    app.config.from_object(config_object)

    # Init extensions
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)

    # Import models to make migrate work
    from app.models.analysis_entity import HealthAnalysis

    # Cloudinary
    cloudinary.config(
        cloud_name=app.config["CLOUDINARY_CLOUD_NAME"],
        api_key=app.config["CLOUDINARY_API_KEY"],
        api_secret=app.config["CLOUDINARY_API_SECRET"],
    )

    register_routes(app)
    from app.controllers.analysis_controller import analysis_blueprint
    app.register_blueprint(analysis_blueprint, url_prefix="/api/analysis")
    return app
