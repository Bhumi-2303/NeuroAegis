from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.database import Base

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(String, primary_key=True, index=True) # UUID string
    name = Column(String, index=True)
    age = Column(Integer)
    gender = Column(String)
    weight = Column(Float)
    height = Column(Float)
    
    # Medical history stored as string to match frontend mock
    medical_history = Column(String, nullable=True)
    
    # Vital signs stored as JSON
    vital_signs = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    status = Column(String, default="active")
    last_visit = Column(DateTime, default=datetime.utcnow)
    
    jobs = relationship("PredictionJob", back_populates="patient")

class PredictionJob(Base):
    __tablename__ = "prediction_jobs"
    
    id = Column(String, primary_key=True, index=True) # UUID string
    patient_id = Column(String, ForeignKey("patients.id"))
    
    status = Column(String, default="Validating") # Current stage
    progress = Column(Integer, default=0) # 0 to 100
    
    # Result data once completed
    prediction_label = Column(String, nullable=True)
    probability_seizure = Column(Float, nullable=True)
    confidence_band = Column(String, nullable=True)
    shap_explanation = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    patient = relationship("Patient", back_populates="jobs")
