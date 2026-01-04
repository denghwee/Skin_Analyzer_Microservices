import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import torch

from app.config import Config

from .service import BaseService

class SkinClassificationService(BaseService):
    def __init__(self):
        self.session = ort.InferenceSession(
            Config.EFFICIENTNET_ONNX_PATH,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

        self.transform = transforms.Compose([
            transforms.Resize(Config.CLASSIFICATION_IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.class_names = Config.CLASS_NAMES

    def _preprocess(self, image):
        """
        Nhận:
        - PIL.Image
        - hoặc FileStorage / file-like object
        Trả về:
        - numpy array shape (1, 3, H, W)
        """

        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        img_tensor = self.transform(image).unsqueeze(0)
        return img_tensor.numpy().astype(np.float32)

    def classify(self, image):
        """
        Classify a single image.
        """

        inp = self._preprocess(image)

        # ONNX inference
        outputs = self.session.run(
            None,
            {self.input_name: inp}
        )

        logits = outputs[0]  # shape (1, NUM_CLASSES)

        # Softmax + argmax
        probs = torch.softmax(torch.from_numpy(logits), dim=1)
        conf, pred = torch.max(probs, 1)

        class_index = int(pred.item())

        return {
            "class_index": class_index,
            "class_name": self.class_names[class_index],
            "confidence": float(conf.item())
        }
