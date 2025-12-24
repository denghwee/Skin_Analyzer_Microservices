from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt

from app.models.analyze_request_dto import AnalyzeRequestDTO
from app.services.analysis_service import AnalysisService

analysis_blueprint = Blueprint("v1/analysis", __name__)


# ====== Helper function lấy userId từ JWT ======
def get_user_id_from_token():
    claims = get_jwt()
    user_id = claims.get("userId")
    if not user_id:
        return None
    return user_id


# ============ API 1: LƯU KẾT QUẢ PHÂN TÍCH SIMPLE =============
@analysis_blueprint.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    """
    API đơn giản lưu kết quả AI khi FE chỉ gửi ít dữ liệu
    """

    user_id = get_user_id_from_token()
    if not user_id:
        return jsonify({"error": "Invalid token — no userId in JWT"}), 401

    data = request.get_json()

    required_fields = ["annotatedImageUrl", "aiDiagnosis", "aiConfidence", "suggestions"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    result = AnalysisService.save_analysis(
        user_id=user_id,
        annotated_url=data["annotatedImageUrl"],
        diagnosis=data["aiDiagnosis"],
        confidence=data["aiConfidence"],
        suggestions=data["suggestions"]
    )

    return jsonify(result.to_json()), 201



# ============ API 2: CẬP NHẬT GHI CHÚ BÁC SĨ =============
@analysis_blueprint.route("/<int:record_id>/doctor-note", methods=["PATCH"])
@jwt_required()
def update_doctor_note(record_id):

    user_id = get_user_id_from_token()
    if not user_id:
        return jsonify({"error": "Invalid token — no userId in JWT"}), 401

    data = request.get_json()

    if "doctorNote" not in data:
        return jsonify({"error": "Missing doctorNote"}), 400

    result = AnalysisService.update_doctor_note(
        record_id=record_id,
        doctor_note=data["doctorNote"],
        user_id=user_id
    )

    if result is None:
        return jsonify({"error": "Record not found or unauthorized"}), 404

    return jsonify(result.to_json()), 200



# ============ API 3: LỊCH SỬ CỦA USER =============
@analysis_blueprint.route("/history", methods=["GET"])
@jwt_required()
def history():

    claims = get_jwt()
    user_id = claims.get("userId")

    if not user_id:
        return jsonify({"error": "Token missing userId"}), 401

    # ⭐ ÉP về int để khớp MySQL
    try:
        user_id = int(user_id)
    except:
        return jsonify({"error": "Invalid userId format"}), 400

    results = AnalysisService.get_history_by_user(user_id)

    return jsonify([r.to_json() for r in results]), 200




# ============ API 4: LƯU FULL RESULT TỪ AI SERVICE ============
@analysis_blueprint.route("/save-ai-result", methods=["POST"])
@jwt_required()
def save_ai_result():

    user_id = get_user_id_from_token()
    if not user_id:
        return jsonify({"error": "Token missing userId in JWT"}), 401

    try:
        dto = AnalyzeRequestDTO(request.get_json())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result = AnalysisService.save_analysis_from_request(user_id, dto)

    return jsonify(result.to_json()), 201
