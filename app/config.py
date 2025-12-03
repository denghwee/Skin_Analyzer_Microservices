import os
import torch

class Config:
    # ======= FLASK / SQLALCHEMY CONFIG =======
    SECRET_KEY = os.getenv("SECRET_KEY")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_ALGORITHM = "HS256"
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Prevent Railway DB timeout
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
        "pool_recycle": 280,
        "pool_timeout": 30,
        "pool_size": 5,
        "max_overflow": 10
    }

    # ======= AI MODEL CONFIG =======
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    EFFICIENTNET_PATH = "app/models/efficientnet_b2_skin.pth"
    YOLO_MODEL_PATH = "app/models/yolov11_skin.pt"

    IMG_SIZE = (224, 224)
    CLASS_NAMES = [
        "none", "acne", "carcinoma", "eczema",
        "keratosis", "rosacea", "milia"
    ]
    NUM_CLASSES = len(CLASS_NAMES)

    CLASSES_REQUIRING_CLASSIFICATION = {
        'acne scar', 'melasma', 'nodules', 'papules',
        'pustules', 'skinredness', 'vascular'
    }

    CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
