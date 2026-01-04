from ultralytics import YOLO
from PIL import Image

from app.config import Config
from .service import BaseService
from app.utils import apply_nms, deduplicate_by_label
from app.utils import draw_boxes, image_to_base64, upload_base64_to_cloudinary

import numpy as np
import onnxruntime as ort
import cv2

class SkinDetectionService(BaseService):
    def __init__(self):
        
        self.session = ort.InferenceSession(
            Config.YOLO_MODEL_ONNX_PATH,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.img_size = (
            Config.DETECTION_IMG_SIZE
            if isinstance(Config.DETECTION_IMG_SIZE, tuple)
            else (int(Config.DETECTION_IMG_SIZE), int(Config.DETECTION_IMG_SIZE))
        )
        
        self.conf_threshold = Config.CONFIDENCE
        self.class_names = Config.CLASS_NAMES  # list[str]

    def preprocess(self, image: Image.Image):
        if not isinstance(image, Image.Image):
            raise TypeError("SkinDetectionService expects PIL.Image.Image")

        # PIL (RGB) → numpy (BGR)
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        h0, w0 = image_np.shape[:2]

        img = cv2.resize(image_np, self.img_size)
        img = img[:, :, ::-1]            # BGR → RGB
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img, (w0, h0)


    def decode(self, outputs, original_size):
        preds = outputs[0][0]  # (N, 6)
        w0, h0 = original_size

        scale_x = w0 / self.img_size[0]
        scale_y = h0 / self.img_size[1]

        detections = []
        for det in preds:
            conf = float(det[4])
            if conf < self.conf_threshold:
                continue

            cls_id = int(det[5])
            x1, y1, x2, y2 = det[:4]

            detections.append({
                "class": self.class_names[cls_id],
                "confidence": conf,
                "bbox": [
                    float(x1 * scale_x),
                    float(y1 * scale_y),
                    float(x2 * scale_x),
                    float(y2 * scale_y),
                ]
            })

        return detections

    def detect(self, image):
        inp, original_size = self.preprocess(image)

        outputs = self.session.run(
            None,
            {self.input_name: inp}
        )

        return self.decode(outputs, original_size)

    def post_process(self, skins):
        skins = apply_nms(skins, iou_threshold=0.5)
        skins = deduplicate_by_label(skins)
        return skins

    def upload_annotated_image(self, image, skins):
        image_with_boxes = draw_boxes(image.copy(), skins)
        base64_img = image_to_base64(image_with_boxes)

        return upload_base64_to_cloudinary(
            base64_img,
            folder="skin-detection/annotated"
        )

    def upload_original_image(self, image):
        return upload_base64_to_cloudinary(
            image,
            folder="skin-detection/original"
        )