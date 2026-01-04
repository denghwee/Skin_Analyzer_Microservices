import os

import torch
from dotenv import load_dotenv
load_dotenv()

class Config:
    DETECTION_SERVICE_URL = os.getenv("DETECTION_SERVICE_URL", "http://localhost:5001/detect")
    CLASSIFICATION_SERVICE_URL = os.getenv(
        "CLASSIFICATION_SERVICE_URL",
        "http://localhost:5002/classify",
    )
    FOOD_DETECTION_SERVICE_URL = os.getenv(
        "FOOD_DETECTION_SERVICE_URL",
        "http://localhost:5003/detect",
    )
    FOOD_CONFIDENCE_THRESHOLD = float(os.getenv("FOOD_CONFIDENCE_THRESHOLD", "0.5"))
    SERVICE_REQUEST_TIMEOUT = float(os.getenv("SERVICE_REQUEST_TIMEOUT", "30"))

    CLASSES_REQUIRING_CLASSIFICATION = set(
        name.strip()
        for name in os.getenv(
            "CLASSES_REQUIRING_CLASSIFICATION",
            "acne scar,melasma,nodules,papules,pustules,skinredness,vascular",
        ).split(",")
        if name.strip()
    )

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

    EFFICIENTNET_PATH = "app/models_AI/efficientnet_b2_skin.pth"
    EFFICIENTNET_ONNX_PATH = "app/models_AI/efficientnet_b2_skin.onnx"

    YOLO_MODEL_PATH = "app/models_AI/yolov10_skin.pt"
    YOLO_MODEL_ONNX_PATH = "app/models_AI/yolov10_skin.onnx"

    DETECTION_IMG_SIZE = (640, 640)
    CLASSIFICATION_IMG_SIZE = (224, 224)
    
    CONFIDENCE = 0.25
    CLASS_NAMES = [
        'None', 'Dark Circle', 'Dark Circle', 'Eyebag', 'Acne Scar', 'Blackhead',
        'Blackheads', 'Dark spot', 'Darkspot', 'Freckle', 'Melasma', 'Nodules',
        'Papules', 'Pustules', 'Skinredness', 'Vascular', 'Whitehead', 'Whiteheads', 'Wrinkle'
    ]
    NUM_CLASSES = len(CLASS_NAMES)

    CLASSES_REQUIRING_CLASSIFICATION = {
        'acne scar', 'melasma', 'nodules', 'papules',
        'pustules', 'skinredness', 'vascular'
    }

    CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
