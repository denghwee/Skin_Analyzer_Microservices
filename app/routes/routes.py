from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt

from app.services_AI.objectdetection import SkinDetectionService
from app.services_AI.classification import SkinClassificationService
from app.utils import (
    crop_regions,
    draw_boxes,
    image_to_base64,
    upload_base64_to_cloudinary,
    generate_health_issue_info,
    generate_lifestyle_suggestions
)
from app.config import Config
from PIL import Image
import io
from datetime import datetime

def register_routes(app):

    @app.route('/')
    def home():
        return "OK", 200

    @app.route('/api/v1/analyze', methods=['POST'])
    @jwt_required()
    def analyze():

        claims = get_jwt()
        user_id = claims.get("userId")
        if not user_id:
            return jsonify({"error": "Invalid token — no userId"}), 401

        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No image uploaded.'}), 400

        image = Image.open(io.BytesIO(file.read())).convert('RGB')

        detect_model = SkinDetectionService()
        detections = detect_model.detect(image)

        if not detections:
            encoded = image_to_base64(image)
            cloud_url = upload_base64_to_cloudinary(encoded)

            return jsonify({
                'status': 'success',
                'annotated_image_url': cloud_url,
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

        # ---- phần dưới giữ nguyên 100% ----
        cropped_images = crop_regions(image, detections)

        results = []
        detection_confidences = []

        for crop, det in zip(cropped_images, detections):
            detected_class = det['class']
            detection_conf = float(det['confidence'])
            detection_confidences.append(detection_conf)

            requires_classification = detected_class in Config.CLASSES_REQUIRING_CLASSIFICATION

            if requires_classification:
                classify_model = SkinClassificationService()
                disease_pred = classify_model.classify(crop)
                results.append({
                    'detected_class': detected_class,
                    'confidence': detection_conf,
                    'bbox': det['bbox'],
                    'disease_prediction': disease_pred,
                    'requires_classification': True
                })
            else:
                results.append({
                    'detected_class': detected_class,
                    'confidence': detection_conf,
                    'bbox': det['bbox'],
                    'disease_prediction': None,
                    'requires_classification': False
                })

        image_with_boxes = draw_boxes(image.copy(), detections)
        annotated_base64 = image_to_base64(image_with_boxes)
        cloud_url = upload_base64_to_cloudinary(annotated_base64)

        health_issue_info = generate_health_issue_info(results, detection_confidences)
        lifestyle_suggestions = generate_lifestyle_suggestions(results, detection_confidences)

        return jsonify({
            'status': 'success',
            'annotated_image_url': cloud_url,
            'detection': results,
            'health_issue_info': health_issue_info,
            'lifestyle_suggestions': lifestyle_suggestions,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_detections': len(results),
                'image_size': {
                    'width': image.width,
                    'height': image.height
                },
                'detection_summary': [
                    {
                        'detected_class': r['detected_class'],
                        'disease': r['disease_prediction'].get('class_name', 'unknown')
                        if r.get('disease_prediction') else None,
                        'detection_confidence': r['confidence'],
                        'classification_confidence': r['disease_prediction'].get('confidence', 0)
                        if r.get('disease_prediction') else None,
                        'requires_classification': r.get('requires_classification', False)
                    }
                    for r in results
                ]
            }
        })

    @app.route('/api/v1/health')
    def health():
        return jsonify({'status': 'ok'})
