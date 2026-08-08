from __future__ import annotations
from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

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
    
    # Privacy & Consent
    consent_given = Column(Boolean, default=False)
    consent_date = Column(DateTime, nullable=True)
    data_retention_opt_in = Column(Boolean, default=False)
    
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
    
    # Dataset detection info
    detected_dataset = Column(String, nullable=True)
    detection_confidence = Column(Float, nullable=True)
    selected_model = Column(String, nullable=True)
    error = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    patient = relationship("Patient", back_populates="jobs")

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, index=True) # UUID string
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="clinician") # admin, clinician, researcher
