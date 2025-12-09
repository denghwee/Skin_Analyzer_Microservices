import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


EFFICIENTNET_PATH = os.getenv("EFFICIENTNET_PATH", "app/models/efficientnet_b2_skin.pth")
INFERENCE_DEVICE = os.getenv("CLASSIFIER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
IMG_HEIGHT = int(os.getenv("CLASSIFIER_IMG_HEIGHT", "224"))
IMG_WIDTH = int(os.getenv("CLASSIFIER_IMG_WIDTH", "224"))
CLASS_NAMES: List[str] = [
    name.strip() for name in os.getenv(
        "CLASS_NAMES",
        "none,acne,carcinoma,eczema,keratosis,rosacea,milia",
    ).split(",") if name.strip()
]
NUM_CLASSES = int(os.getenv("NUM_CLASSES", str(len(CLASS_NAMES))))

_MODEL = models.efficientnet_b2(weights=None)
_NUM_FEATURES = _MODEL.classifier[1].in_features
_MODEL.classifier[1] = nn.Linear(_NUM_FEATURES, NUM_CLASSES)

_STATE_DICT = torch.load(EFFICIENTNET_PATH, map_location=INFERENCE_DEVICE)
if "net" in _STATE_DICT:
    _STATE_DICT = _STATE_DICT["net"]

_MODEL.load_state_dict(_STATE_DICT)
_MODEL.to(INFERENCE_DEVICE)
_MODEL.eval()

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def classify_image(image_input: Image.Image) -> Dict[str, Any]:
    if not isinstance(image_input, Image.Image):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    img_tensor = _TRANSFORM(image).unsqueeze(0).to(INFERENCE_DEVICE)

    with torch.no_grad():
        outputs = _MODEL(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    class_index = int(pred.item())
    class_name = CLASS_NAMES[class_index] if class_index < len(CLASS_NAMES) else str(class_index)

    return {
        "class_index": class_index,
        "class_name": class_name,
        "confidence": float(conf.item()),
    }

