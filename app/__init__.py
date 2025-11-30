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
def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Load config (MySQL, secret key…)
    app.config.from_object(Config)


    # Init database + migrate
    db.init_app(app)
    migrate.init_app(app, db)

    # ⭐⭐⭐ QUAN TRỌNG: IMPORT ENTITY TẠI ĐÂY ⭐⭐⭐
    # Nếu không import, Migrate sẽ không thấy model => "No changes detected"
    from app.models.analysis_entity import HealthAnalysis
    cloudinary.config(
        cloud_name=app.config["CLOUDINARY_CLOUD_NAME"],
        api_key=app.config["CLOUDINARY_API_KEY"],
        api_secret=app.config["CLOUDINARY_API_SECRET"]
    )
    # Register blueprints (route groups)
    register_routes(app)
    jwt.init_app(app)
    return app
