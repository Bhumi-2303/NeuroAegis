from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import uuid
from datetime import datetime

from app.db.database import get_db
from app.db.models import Patient
from app.schemas.patient import PatientCreate, PatientResponse

router = APIRouter()

@router.post("/", response_model=PatientResponse, status_code=201)
def create_patient(patient_in: PatientCreate, db: Session = Depends(get_db)):
    db_patient = Patient(
        id=str(uuid.uuid4()),
        name=patient_in.name,
        age=patient_in.age,
        gender=patient_in.gender,
        weight=patient_in.weight,
        height=patient_in.height,
        medical_history=patient_in.medical_history,
        vital_signs=patient_in.vital_signs,
        created_at=datetime.utcnow()
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

@router.get("/", response_model=List[PatientResponse])
def get_patients(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    patients = db.query(Patient).offset(skip).limit(limit).all()
    return patients

@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient
