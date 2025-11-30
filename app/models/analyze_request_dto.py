class AnalyzeRequestDTO:

    def __init__(self, data: dict):
        """
        DTO dùng để validate và parse request cho analyze API
        """

        # ===== REQUIRED FIELDS =====
        required_fields = [
            "annotated_image_url",
            "detection",
            "health_issue_info",
            "lifestyle_suggestions",
            "metadata",
            "status"
        ]

        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"Missing fields: {missing}")

        # ===== ASSIGN VALUES =====
        self.annotated_image_url = data["annotated_image_url"]
        self.detection = data["detection"]  # list
        self.health_issue_info = data.get("health_issue_info")
        self.lifestyle_suggestions = data["lifestyle_suggestions"]  # dict
        self.metadata = data["metadata"]  # dict
        self.status = data["status"]

        # ===== EXTRA VALIDATION =====
        if not isinstance(self.detection, list):
            raise ValueError("Field 'detection' must be a list")

        if not isinstance(self.lifestyle_suggestions, dict):
            raise ValueError("Field 'lifestyle_suggestions' must be a dict")

        if not isinstance(self.metadata, dict):
            raise ValueError("Field 'metadata' must be a dict")

    def to_json(self):
        """
        Trả về dạng JSON đúng chuẩn để dùng lại nếu cần.
        """
        return {
            "annotated_image_url": self.annotated_image_url,
            "detection": self.detection,
            "health_issue_info": self.health_issue_info,
            "lifestyle_suggestions": self.lifestyle_suggestions,
            "metadata": self.metadata,
            "status": self.status,
        }
