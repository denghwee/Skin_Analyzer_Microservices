from flask import request, jsonify, render_template
from .objectdetection_service import detect_objects
from .classification_service import classify_image
from .utils import crop_regions, draw_boxes, image_to_base64
from .health_info import generate_health_issue_info, generate_lifestyle_suggestions
from .config import Config
from PIL import Image
import io
from datetime import datetime

def register_routes(app):

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/analyze', methods=['POST'])
    def analyze():
        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No image uploaded.'}), 400

        image = Image.open(io.BytesIO(file.read())).convert('RGB')

        detections = detect_objects(image)
        if not detections:
            return jsonify({
                'status': 'success',
                'annotated_image_base64': image_to_base64(image),
                'detection': [],
                'health_issue_info': None,
                'lifestyle_suggestions': {
                    'lifestyle': ['Không phát hiện vấn đề cụ thể. Tiếp tục chăm sóc da hàng ngày.'],
                    'diet': ['Duy trì chế độ ăn cân bằng và lành mạnh.']
                },
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_detections': 0,
                    'image_size': {
                        'width': image.width,
                        'height': image.height
                    },
                    'detection_summary': []
                }
            })

        cropped_images = crop_regions(image, detections)

        results = []
        detection_confidences = []
        for crop, det in zip(cropped_images, detections):
            detected_class = det['class']
            detection_conf = float(det['confidence'])
            detection_confidences.append(detection_conf)
            
            # Chỉ đưa vào classification nếu là bệnh da liễu thực sự
            requires_classification = detected_class in Config.CLASSES_REQUIRING_CLASSIFICATION
            
            if requires_classification:
                # Gọi classification service cho các bệnh da liễu
                disease_pred = classify_image(crop)
                results.append({
                    'detected_class': detected_class,
                    'confidence': detection_conf,
                    'bbox': det['bbox'],
                    'disease_prediction': disease_pred,
                    'requires_classification': True
                })
            else:
                # Chỉ có detection, không cần classification
                results.append({
                    'detected_class': detected_class,
                    'confidence': detection_conf,
                    'bbox': det['bbox'],
                    'disease_prediction': None,
                    'requires_classification': False
                })

        image_with_boxes = draw_boxes(image.copy(), detections)
        encoded_img = image_to_base64(image_with_boxes)

        # Tạo thông tin sức khỏe và gợi ý
        health_issue_info = generate_health_issue_info(results, detection_confidences)
        lifestyle_suggestions = generate_lifestyle_suggestions(results, detection_confidences)

        # Metadata cho LLM + RAG
        timestamp = datetime.now().isoformat()
        metadata = {
            'timestamp': timestamp,
            'total_detections': len(results),
            'image_size': {
                'width': image.width,
                'height': image.height
            },
            'detection_summary': [
                {
                    'detected_class': r['detected_class'],
                    'disease': r['disease_prediction'].get('class_name', 'unknown') if r.get('disease_prediction') else None,
                    'detection_confidence': r['confidence'],
                    'classification_confidence': r['disease_prediction'].get('confidence', 0) if r.get('disease_prediction') else None,
                    'requires_classification': r.get('requires_classification', False)
                }
                for r in results
            ]
        }

        return jsonify({
            'status': 'success',
            'annotated_image_base64': encoded_img,
            'detection': results,
            'health_issue_info': health_issue_info,
            'lifestyle_suggestions': lifestyle_suggestions,
            'metadata': metadata
        })

    @app.route('/health')
    def health():
        return jsonify({'status': 'ok'})
