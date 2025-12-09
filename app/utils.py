import base64
from io import BytesIO
from typing import List, Dict, Any
from PIL import Image, ImageDraw

def crop_regions(image, detections):
    crops = []
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        crop = image.crop((x1, y1, x2, y2))
        crops.append(crop)
    return crops

def draw_boxes(image, detections):
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, y1 - 10), f"{det['class']} ({det['confidence']:.2f})", fill="red")
    return image

def image_to_base64(image, format="JPEG"):
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    return encoded


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        float: IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def apply_nms(detections: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Apply Non-Maximum Suppression to remove duplicate detections.
    
    Merges ALL detections with the same class that have overlapping bounding boxes (IoU > threshold).
    Keeps the detection with highest confidence and merges all overlapping boxes together.
    
    Args:
        detections: List of detection dicts with 'class', 'confidence', 'bbox'
        iou_threshold: IoU threshold for considering boxes as overlapping (default 0.5)
    
    Returns:
        List of deduplicated detections with merged boxes
    """
    if not detections:
        return []
    
    # Group detections by class
    class_groups = {}
    for det in detections:
        class_name = det['class']
        if class_name not in class_groups:
            class_groups[class_name] = []
        class_groups[class_name].append(det)
    
    # Apply NMS within each class group - merge ALL overlapping detections
    merged_detections = []
    
    for class_name, class_dets in class_groups.items():
        if not class_dets:
            continue
        
        # Sort by confidence (descending)
        class_dets = sorted(class_dets, key=lambda x: -float(x['confidence']))
        
        # Use a cluster-based approach: group all overlapping detections together
        clusters = []  # Each cluster is a list of detection indices that overlap
        
        for i, det_i in enumerate(class_dets):
            # Find which cluster this detection belongs to
            found_cluster = False
            
            for cluster in clusters:
                # Check if this detection overlaps with ANY detection in the cluster
                for j in cluster:
                    det_j = class_dets[j]
                    iou = calculate_iou(det_i['bbox'], det_j['bbox'])
                    
                    if iou > iou_threshold:
                        cluster.append(i)
                        found_cluster = True
                        break
                
                if found_cluster:
                    break
            
            # If not found in any cluster, create a new cluster
            if not found_cluster:
                clusters.append([i])
        
        # Merge each cluster into a single detection
        for cluster in clusters:
            if not cluster:
                continue
            
            # Get the detection with highest confidence in this cluster
            cluster_dets = [class_dets[i] for i in cluster]
            best_det = max(cluster_dets, key=lambda x: float(x['confidence']))
            
            # Merge all bounding boxes in the cluster (take union)
            all_boxes = [det['bbox'] for det in cluster_dets]
            
            x1_min = min(box[0] for box in all_boxes)
            y1_min = min(box[1] for box in all_boxes)
            x1_max = max(box[2] for box in all_boxes)
            y1_max = max(box[3] for box in all_boxes)
            
            # Create merged detection
            merged_det = best_det.copy()
            merged_det['bbox'] = [x1_min, y1_min, x1_max, y1_max]
            
            merged_detections.append(merged_det)
    
    return merged_detections


def deduplicate_by_label(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Final deduplication: keep only the highest confidence detection for each unique label.
    
    After NMS merges overlapping boxes of the same class, this step ensures
    only ONE detection per unique class label in the final output.
    
    Args:
        detections: List of detection dicts with 'class', 'confidence', 'bbox'
    
    Returns:
        List of deduplicated detections (max 1 per unique label)
    """
    if not detections:
        return []
    
    # Group by class label and keep only the highest confidence one
    label_best = {}
    
    for det in detections:
        class_name = det['class']
        
        if class_name not in label_best:
            label_best[class_name] = det
        else:
            # Keep the one with higher confidence
            if float(det['confidence']) > float(label_best[class_name]['confidence']):
                label_best[class_name] = det
    
    # Return as list, sorted by confidence descending
    result = list(label_best.values())
    result = sorted(result, key=lambda x: -float(x['confidence']))
    
    return result