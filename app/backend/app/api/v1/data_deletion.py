from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.audit import EVENT_CLINICAL_PATIENT_DELETED, record_audit_event
from app.core.auth import get_tenant_patient, require_role
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
    Permanently deletes a patient record and all associated prediction jobs within the admin's tenant.
    This fulfills GDPR Article 17 (Right to Erasure / Right to be Forgotten).
    Only admins within the authoritative tenant can perform data deletion.
    """
    patient = get_tenant_patient(patient_id, db, current_user.tenant_id, actor_id=current_user.id)

    # Insert audit event in the same transaction; survives patient deletion
    record_audit_event(
        db=db,
        event_type=EVENT_CLINICAL_PATIENT_DELETED,
        outcome="success",
        actor_type="user",
        actor_id=current_user.id,
        tenant_id=current_user.tenant_id,
        patient_id=patient.id,
        resource_type="patient",
        resource_id=patient.id,
    )

    # Delete all associated prediction jobs for this patient within the authoritative tenant
    db.query(PredictionJob).filter(
        PredictionJob.patient_id == patient_id,
        PredictionJob.tenant_id == current_user.tenant_id,
    ).delete()
    db.delete(patient)
    db.commit()

    return {
        "status": "deleted",
        "patient_id": patient_id,
        "message": "Patient record and all associated data have been permanently deleted.",
    }
