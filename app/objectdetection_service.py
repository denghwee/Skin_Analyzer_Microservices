from ultralytics import YOLO
from .config import Config

model = YOLO(Config.YOLO_MODEL_PATH)

def detect_objects(image):
    results = model.predict(image, conf=0.25, imgsz=640)
    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": model.names[int(box.cls)],
            "confidence": float(box.conf),
            "bbox": [float(x) for x in box.xyxy[0].tolist()]
        })
    return detections