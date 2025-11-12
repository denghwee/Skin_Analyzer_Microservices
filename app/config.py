import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    EFFICIENTNET_PATH = "app/models/efficientnet_b2_skin.pth"
    YOLO_MODEL_PATH = "app/models/yolov11_skin.pt"

    IMG_SIZE = (224, 224)
    CLASS_NAMES = ["none", "acne", "carcinoma", "eczema", "keratosis", "rosacea", "milia"]
    NUM_CLASSES = len(CLASS_NAMES)
    
    CLASSES_REQUIRING_CLASSIFICATION = {
        'acne scar', 'melasma', 'nodules', 'papules',
        'pustules', 'skinredness', 'vascular'
    }