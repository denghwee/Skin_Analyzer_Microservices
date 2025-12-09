import os


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
