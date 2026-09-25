from __future__ import annotations
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    event,
)
from sqlalchemy.orm import Session, relationship, validates

from app.db.database import Base

DEFAULT_TENANT_ID = "00000000-0000-0000-0000-000000000001"
DEFAULT_TENANT_SLUG = "default-org"
DEFAULT_TENANT_NAME = "Default Organization"


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(String, primary_key=True, index=True)  # UUID string
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    users = relationship("User", back_populates="tenant", foreign_keys="User.tenant_id")
    patients = relationship("Patient", back_populates="tenant", foreign_keys="Patient.tenant_id")
    prediction_jobs = relationship("PredictionJob", back_populates="tenant", foreign_keys="PredictionJob.tenant_id")

    @validates("slug")
    def validate_slug(self, key, slug):
        if not slug or not str(slug).strip():
            raise ValueError("Tenant slug cannot be empty")
        return str(slug).strip().lower()

    @validates("name")
    def validate_name(self, key, name):
        if not name or not str(name).strip():
            raise ValueError("Tenant name cannot be empty")
        return str(name).strip()


class Patient(Base):
    __tablename__ = "patients"

    id = Column(String, primary_key=True, index=True)  # UUID string
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

    # Tenancy & Ownership (Prompt 9.1)
    tenant_id = Column(String, ForeignKey("tenants.id"), index=True, nullable=False, default=DEFAULT_TENANT_ID)
    created_by_user_id = Column(String, ForeignKey("users.id"), index=True, nullable=True)
    is_deleted = Column(Boolean, default=False, nullable=False)

    tenant = relationship("Tenant", back_populates="patients", foreign_keys=[tenant_id])
    created_by = relationship("User", back_populates="created_patients", foreign_keys=[created_by_user_id])
    jobs = relationship("PredictionJob", back_populates="patient", foreign_keys="PredictionJob.patient_id")

    @validates("tenant_id")
    def validate_tenant_id(self, key, tenant_id):
        if not tenant_id or not str(tenant_id).strip():
            raise ValueError("Patient tenant_id cannot be empty")
        return str(tenant_id).strip()

    @validates("created_by")
    def validate_created_by_tenant(self, key, created_by):
        if created_by is not None and self.tenant_id is not None and getattr(created_by, "tenant_id", None) is not None:
            if self.tenant_id != created_by.tenant_id:
                raise ValueError(
                    f"Tenant consistency violation: Patient tenant '{self.tenant_id}' "
                    f"does not match creator User tenant '{created_by.tenant_id}'"
                )
        return created_by


class PredictionJob(Base):
    __tablename__ = "prediction_jobs"

    id = Column(String, primary_key=True, index=True)  # UUID string
    patient_id = Column(String, ForeignKey("patients.id"), index=True, nullable=True)

    status = Column(String, default="Validating")  # Current stage
    progress = Column(Integer, default=0)  # 0 to 100

    # Result data once completed
    prediction_label = Column(String, nullable=True)
    probability_seizure = Column(Float, nullable=True)
    confidence_band = Column(String, nullable=True)
    shap_explanation = Column(JSON, nullable=True)
    eeg_visualization = Column(JSON, nullable=True)

    # Dataset detection info
    detected_dataset = Column(String, nullable=True)
    detection_confidence = Column(Float, nullable=True)
    selected_model = Column(String, nullable=True)
    error = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Worker Lifecycle & Leases (Prompt 7.4)
    worker_id = Column(String, nullable=True, index=True)
    heartbeat_at = Column(DateTime, nullable=True)
    lease_expires_at = Column(DateTime, nullable=True, index=True)

    # Tenancy & Ownership (Prompt 9.1)
    tenant_id = Column(String, ForeignKey("tenants.id"), index=True, nullable=False, default=DEFAULT_TENANT_ID)
    created_by_user_id = Column(String, ForeignKey("users.id"), index=True, nullable=True)
    is_deleted = Column(Boolean, default=False, nullable=False)

    tenant = relationship("Tenant", back_populates="prediction_jobs", foreign_keys=[tenant_id])
    created_by = relationship("User", back_populates="created_prediction_jobs", foreign_keys=[created_by_user_id])
    patient = relationship("Patient", back_populates="jobs", foreign_keys=[patient_id])

    @validates("tenant_id")
    def validate_tenant_id(self, key, tenant_id):
        if not tenant_id or not str(tenant_id).strip():
            raise ValueError("PredictionJob tenant_id cannot be empty")
        return str(tenant_id).strip()

    @validates("patient")
    def validate_patient_tenant(self, key, patient):
        if patient is not None and self.tenant_id is not None and getattr(patient, "tenant_id", None) is not None:
            if self.tenant_id != patient.tenant_id:
                raise ValueError(
                    f"Tenant consistency violation: PredictionJob tenant '{self.tenant_id}' "
                    f"does not match Patient tenant '{patient.tenant_id}'"
                )
        return patient

    @validates("created_by")
    def validate_created_by_tenant(self, key, created_by):
        if created_by is not None and self.tenant_id is not None and getattr(created_by, "tenant_id", None) is not None:
            if self.tenant_id != created_by.tenant_id:
                raise ValueError(
                    f"Tenant consistency violation: PredictionJob tenant '{self.tenant_id}' "
                    f"does not match creator User tenant '{created_by.tenant_id}'"
                )
        return created_by


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)  # UUID string
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="clinician", nullable=False)  # admin, clinician, researcher
    tenant_id = Column(String, ForeignKey("tenants.id"), index=True, nullable=True, default=DEFAULT_TENANT_ID)
    is_active = Column(Boolean, default=True, nullable=False)
    token_version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    tenant = relationship("Tenant", back_populates="users", foreign_keys=[tenant_id])
    created_patients = relationship("Patient", back_populates="created_by", foreign_keys="Patient.created_by_user_id")
    created_prediction_jobs = relationship("PredictionJob", back_populates="created_by", foreign_keys="PredictionJob.created_by_user_id")

    @validates("role")
    def validate_role(self, key, role):
        valid_roles = {"admin", "clinician", "researcher"}
        if role not in valid_roles:
            raise ValueError(f"Invalid user role: {role}. Must be one of {valid_roles}")
        return role


@event.listens_for(Session, "before_flush")
def validate_tenant_consistency(session, flush_context, instances):
    for obj in session.new.union(session.dirty):
        if isinstance(obj, PredictionJob):
            if obj.patient is not None and obj.tenant_id and obj.patient.tenant_id:
                if obj.tenant_id != obj.patient.tenant_id:
                    raise ValueError(
                        f"Tenant consistency violation: PredictionJob tenant '{obj.tenant_id}' "
                        f"does not match Patient tenant '{obj.patient.tenant_id}'"
                    )
            if obj.created_by is not None and obj.tenant_id and obj.created_by.tenant_id:
                if obj.tenant_id != obj.created_by.tenant_id:
                    raise ValueError(
                        f"Tenant consistency violation: PredictionJob tenant '{obj.tenant_id}' "
                        f"does not match creator User tenant '{obj.created_by.tenant_id}'"
                    )
        elif isinstance(obj, Patient):
            if obj.created_by is not None and obj.tenant_id and obj.created_by.tenant_id:
                if obj.tenant_id != obj.created_by.tenant_id:
                    raise ValueError(
                        f"Tenant consistency violation: Patient tenant '{obj.tenant_id}' "
                        f"does not match creator User tenant '{obj.created_by.tenant_id}'"
                    )
