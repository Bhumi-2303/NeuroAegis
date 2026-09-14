from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.auth import require_role
from app.db.database import get_db
from app.db.models import Patient, PredictionJob, User

router = APIRouter()


@router.delete("/patient/{patient_id}")
def delete_patient_data(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """
    Permanently deletes a patient record and all associated prediction jobs.
    This fulfills GDPR Article 17 (Right to Erasure / Right to be Forgotten).
    Only admins can perform data deletion.
    """
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Delete all associated prediction jobs first (foreign key constraint)
    db.query(PredictionJob).filter(PredictionJob.patient_id == patient_id).delete()
    db.delete(patient)
    db.commit()

    return {
        "status": "deleted",
        "patient_id": patient_id,
        "message": "Patient record and all associated data have been permanently deleted.",
    }
