from datetime import datetime
import io

from flask import jsonify, render_template, request
from PIL import Image

from .config import Config
from .food_nutrition import calculate_total_nutrition
from .health_info import generate_health_issue_info, generate_lifestyle_suggestions
from .config import Config
from PIL import Image
import io
from datetime import datetime
from app.utils import upload_base64_to_cloudinary
from app.config import Config
from .service_clients import (
    ServiceCallError,
    call_classification_service,
    call_detection_service,
    call_food_detection_service,
)
from .utils import crop_regions, draw_boxes, image_to_base64, apply_nms, deduplicate_by_label

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

        try:
            detections = call_detection_service(image)
        except ServiceCallError as error:
            return jsonify({'error': str(error)}), 502
        
        # Apply NMS to remove duplicate detections (same class + overlapping boxes)
        detections = apply_nms(detections, iou_threshold=0.5)
        
        # Final deduplication: keep only highest confidence per unique label
        detections = deduplicate_by_label(detections)
        
        if not detections:
            # No detection → VẪN UPLOAD ẢNH GỐC lên Cloudinary
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

        # Có detection → xử lý crop & classification
        cropped_images = crop_regions(image, detections)

        results = []
        detection_confidences = []

        for crop, det in zip(cropped_images, detections):
            detected_class = det['class']
            detection_conf = float(det['confidence'])
            detection_confidences.append(detection_conf)

            requires_classification = detected_class in Config.CLASSES_REQUIRING_CLASSIFICATION

            # Nếu là bệnh thật → gọi classification
            if requires_classification:
                disease_pred = classify_image(crop)
                # Gọi classification service cho các bệnh da liễu
                try:
                    disease_pred = call_classification_service(crop)
                except ServiceCallError as error:
                    disease_pred = {
                        'class_index': None,
                        'class_name': None,
                        'confidence': 0,
                        'error': str(error)
                    }
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

        # Vẽ bounding boxes lên ảnh
        image_with_boxes = draw_boxes(image.copy(), detections)

        # Convert annotated image → base64
        annotated_base64 = image_to_base64(image_with_boxes)

        # UPLOAD TO CLOUDINARY 🚀
        cloud_url = upload_base64_to_cloudinary(annotated_base64)

        # Health info + suggestions
        health_issue_info = generate_health_issue_info(results, detection_confidences)
        lifestyle_suggestions = generate_lifestyle_suggestions(results, detection_confidences)

        # Metadata
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
                    'disease': r['disease_prediction'].get('class_name', 'unknown') if r.get(
                        'disease_prediction') else None,
                    'detection_confidence': r['confidence'],
                    'classification_confidence': r['disease_prediction'].get('confidence', 0) if r.get(
                        'disease_prediction') else None,
                    'requires_classification': r.get('requires_classification', False)
                }
                for r in results
            ]
        }

        return jsonify({
            'status': 'success',
            'annotated_image_url': cloud_url,  # ⭐⭐⭐ KHÔNG GỬI BASE64 NỮA
            'detection': results,
            'health_issue_info': health_issue_info,
            'lifestyle_suggestions': lifestyle_suggestions,
            'metadata': metadata
        })

    @app.route('/health')
    def health():
        return jsonify({'status': 'ok'})

    @app.route('/detect_food', methods=['POST'])
    def detect_food():
        """Endpoint for separate food-detection pipeline.

        Returns all food items above configured threshold.
        Format matches /analyze endpoint for consistency.
        """
        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No image uploaded.'}), 400

        image = Image.open(io.BytesIO(file.read())).convert('RGB')

        try:
            foods = call_food_detection_service(image)
        except ServiceCallError as error:
            return jsonify({'error': str(error)}), 502

        # Normalize structure: expect list of {'class': name, 'confidence': float, 'bbox': [...]}
        foods = foods or []
        
        # Apply NMS to remove duplicate detections (same food + overlapping boxes)
        foods = apply_nms(foods, iou_threshold=0.5)
        
        # Final deduplication: keep only highest confidence per unique label
        foods = deduplicate_by_label(foods)

        # Filter by configured threshold and create results
        threshold = Config.FOOD_CONFIDENCE_THRESHOLD
        results = [
            {
                'detected_class': f.get('class', 'unknown'),
                'confidence': float(f.get('confidence', 0)),
                'bbox': f.get('bbox', []),
                'disease_prediction': None,  # No disease classification for foods
                'requires_classification': False
            }
            for f in foods
            if float(f.get('confidence', 0)) >= threshold
        ]

        if not results:
            return jsonify({
                'status': 'success',
                'annotated_image_base64': image_to_base64(image),
                'detection': [],
                'health_issue_info': None,
                'nutrition_analysis': {
                    'individual_items': [],
                    'total_nutrition': {
                        'Calories': 0,
                        'Fat': 0,
                        'Saturates': 0,
                        'Sugar': 0,
                        'Salt': 0
                    },
                    'items_count': 0
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

        # Annotated image with boxes
        image_with_boxes = draw_boxes(image.copy(), foods)
        encoded_img = image_to_base64(image_with_boxes)

        # Create detection summary
        detection_summary = [
            {
                'detected_class': r['detected_class'],
                'disease': None,
                'detection_confidence': r['confidence'],
                'classification_confidence': None,
                'requires_classification': False
            }
            for r in results
        ]
        
        # Calculate nutrition analysis
        nutrition_analysis = calculate_total_nutrition(results)

        return jsonify({
            'status': 'success',
            'annotated_image_base64': encoded_img,
            'detection': results,
            'health_issue_info': None,
            'nutrition_analysis': nutrition_analysis,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_detections': len(results),
                'image_size': {
                    'width': image.width,
                    'height': image.height
                },
                'detection_summary': detection_summary,
                'threshold': threshold
            }
        })

    from app.controllers.analysis_controller import analysis_blueprint
    app.register_blueprint(analysis_blueprint, url_prefix="/api/analysis")
