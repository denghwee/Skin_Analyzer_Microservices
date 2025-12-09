from app import db
from app.models.analysis_entity import HealthAnalysis
from app.models.analysis_model import AnalysisResult
from datetime import datetime

class AnalysisService:

    @staticmethod
    def save_analysis_from_request(user_id, dto):
        """
        Lưu kết quả phân tích từ DTO request (AnalyzeRequestDTO)
        """

        # ==== 1. LẤY DỮ LIỆU TỪ DTO ====
        annotated_url = dto.annotated_image_url

        # Detection luôn là list → lấy cái đầu tiên
        first_det = dto.detection[0] if len(dto.detection) > 0 else None

        ai_diagnosis = first_det["detected_class"] if first_det else None
        ai_confidence = first_det["confidence"] if first_det else None

        # Suggestions là 2 nhóm: diet + lifestyle
        suggestions = dto.lifestyle_suggestions

        # ==== 2. TẠO ENTITY ====
        entity = HealthAnalysis(
            user_id=user_id,
            analysis_image_url=annotated_url,
            ai_diagnosis=ai_diagnosis,
            ai_confidence=ai_confidence,
            suggestions=suggestions,
            created_at=datetime.now()
        )

        # ==== 3. LƯU DB ====
        db.session.add(entity)
        db.session.commit()

        # ==== 4. TRẢ DTO RESPONSE ====
        return AnalysisResult(entity)


    @staticmethod
    def update_doctor_note(record_id, doctor_note, user_id):

        entity = HealthAnalysis.query.get(record_id)
        if not entity:
            return None

        # Chỉ cho phép user sửa record của chính họ
        if entity.user_id != user_id:
            return None

        entity.doctor_note = doctor_note
        entity.doctor_updated_at = datetime.now()

        db.session.commit()

        return AnalysisResult(entity)


    @staticmethod
    def get_history_by_user(user_id):

        records = (
            HealthAnalysis.query
            .filter_by(user_id= user_id)
            .order_by(HealthAnalysis.created_at.desc())
            .all()
        )

        return [AnalysisResult(record) for record in records]
