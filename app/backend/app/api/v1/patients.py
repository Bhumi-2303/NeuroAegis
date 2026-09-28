from __future__ import annotations
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.auth import get_tenant_patient, require_roles
from app.db.database import get_db
from app.db.models import Patient, User
from app.schemas.patient import PatientCreate, PatientResponse

router = APIRouter()


@router.post("/", response_model=PatientResponse, status_code=201)
def create_patient(
    patient_in: PatientCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("clinician", "admin")),
):
    db_patient = Patient(
        id=str(uuid.uuid4()),
        name=patient_in.name,
        age=patient_in.age,
        gender=patient_in.gender,
        weight=patient_in.weight,
        height=patient_in.height,
        medical_history=patient_in.medical_history,
        vital_signs=patient_in.vital_signs,
        tenant_id=current_user.tenant_id,
        created_by_user_id=current_user.id,
        created_at=datetime.utcnow(),
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient


@router.get("/", response_model=list[PatientResponse])
def get_patients(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("clinician", "admin")),
):
    patients = (
        db.query(Patient)
        .filter(
            Patient.tenant_id == current_user.tenant_id,
            Patient.is_deleted == False,
        )
        .order_by(Patient.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return patients


@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("clinician", "admin")),
):
    return get_tenant_patient(patient_id, db, current_user.tenant_id)

