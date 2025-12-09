class AnalysisResult:
    """
    DTO trả về FE, chuyển từ Entity → JSON.
    Không liên quan đến DB.
    """

    def __init__(self, entity):
        self.id = entity.id
        self.analysisImageUrl = entity.analysis_image_url
        self.aiDiagnosis = entity.ai_diagnosis
        self.aiConfidence = entity.ai_confidence
        self.suggestions = entity.suggestions
        self.doctorNote = entity.doctor_note
        self.createdAt = entity.created_at.isoformat()

    def to_json(self):
        return {
            "id": self.id,
            "analysisImageUrl": self.analysisImageUrl,
            "aiDiagnosis": self.aiDiagnosis,
            "aiConfidence": self.aiConfidence,
            "suggestions": self.suggestions,
            "doctorNote": self.doctorNote,
            "createdAt": self.createdAt
        }
