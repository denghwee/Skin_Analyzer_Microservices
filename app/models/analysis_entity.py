from app import db
from datetime import datetime

class HealthAnalysis(db.Model):
    __tablename__ = "health_analysis"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)

    analysis_image_url = db.Column(db.String(500), nullable=False)
    ai_diagnosis = db.Column(db.String(255), nullable=False)
    ai_confidence = db.Column(db.Float, nullable=False)

    suggestions = db.Column(db.JSON, nullable=False)
    doctor_note = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.now)
    doctor_updated_at = db.Column(db.DateTime, nullable=True)

    def __repr__(self):
        return f"<HealthAnalysis id={self.id}, user={self.user_id}>"
