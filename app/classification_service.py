from torchvision import models, transforms
import torch.nn as nn
from .config import Config
import torch
from PIL import Image

model = models.efficientnet_b2(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, Config.NUM_CLASSES)

state_dict = torch.load(Config.EFFICIENTNET_PATH, map_location=Config.DEVICE)
if "net" in state_dict:
    state_dict = state_dict["net"]

model.load_state_dict(state_dict)
model.to(Config.DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize(Config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def classify_image(image_input):
    """
    Nhận input là:
    - file ảnh (FileStorage từ Flask)
    - hoặc đối tượng PIL.Image (vùng crop từ YOLO)
    Trả về label class và confidence
    """
    # Xử lý input
    if not isinstance(image_input, Image.Image):
        image = Image.open(image_input).convert('RGB')
    else:
        image = image_input.convert('RGB')

    img_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return {
        "class_index": int(pred.item()),
        "class_name": Config.CLASS_NAMES[int(pred.item())],
        "confidence": float(conf.item())
    }
