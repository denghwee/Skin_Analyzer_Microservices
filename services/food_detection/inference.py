import os
from typing import Any, Dict, List

from ultralytics import YOLO


YOLO_MODEL_PATH = os.getenv("FOOD_YOLO_MODEL_PATH", "app/models/food_detection.pt")
YOLO_CONFIDENCE = float(os.getenv("FOOD_YOLO_CONFIDENCE", "0.25"))
YOLO_IMAGE_SIZE = int(os.getenv("FOOD_YOLO_IMAGE_SIZE", "640"))
YOLO_DEVICE = os.getenv("FOOD_YOLO_DEVICE", "cpu")
NMS_IOU_THRESHOLD = float(os.getenv("NMS_IOU_THRESHOLD", "0.5"))

_MODEL = YOLO(YOLO_MODEL_PATH)
_CLASS_NAMES = getattr(_MODEL, "names", {})


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def apply_nms(detections: List[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
    """Apply NMS to merge overlapping detections with same class."""
    if not detections:
        return []
    
    # Group detections by class
    class_groups = {}
    for det in detections:
        class_name = det['class']
        if class_name not in class_groups:
            class_groups[class_name] = []
        class_groups[class_name].append(det)
    
    merged_detections = []
    
    for class_name, class_dets in class_groups.items():
        if not class_dets:
            continue
        
        # Sort by confidence (descending)
        class_dets = sorted(class_dets, key=lambda x: -float(x['confidence']))
        
        # Cluster overlapping detections
        clusters = []
        
        for i, det_i in enumerate(class_dets):
            found_cluster = False
            
            for cluster in clusters:
                for j in cluster:
                    det_j = class_dets[j]
                    iou = calculate_iou(det_i['bbox'], det_j['bbox'])
                    
                    if iou > iou_threshold:
                        cluster.append(i)
                        found_cluster = True
                        break
                
                if found_cluster:
                    break
            
            if not found_cluster:
                clusters.append([i])
        
        # Merge each cluster
        for cluster in clusters:
            if not cluster:
                continue
            
            cluster_dets = [class_dets[i] for i in cluster]
            best_det = max(cluster_dets, key=lambda x: float(x['confidence']))
            
            all_boxes = [det['bbox'] for det in cluster_dets]
            x1_min = min(box[0] for box in all_boxes)
            y1_min = min(box[1] for box in all_boxes)
            x1_max = max(box[2] for box in all_boxes)
            y1_max = max(box[3] for box in all_boxes)
            
            merged_det = best_det.copy()
            merged_det['bbox'] = [x1_min, y1_min, x1_max, y1_max]
            
            merged_detections.append(merged_det)
    
    return merged_detections


def detect_foods(image) -> List[Dict[str, Any]]:
    results = _MODEL.predict(
        image,
        conf=YOLO_CONFIDENCE,
        imgsz=YOLO_IMAGE_SIZE,
        device=YOLO_DEVICE,
        verbose=False,
    )
    detections: List[Dict[str, Any]] = []
    for box in results[0].boxes:
        detections.append(
            {
                "class": _CLASS_NAMES.get(int(box.cls), str(int(box.cls))),
                "confidence": float(box.conf),
                "bbox": [float(x) for x in box.xyxy[0].tolist()],
            }
        )
    
    # Apply NMS to remove duplicates
    detections = apply_nms(detections, NMS_IOU_THRESHOLD)
    
    return detections
