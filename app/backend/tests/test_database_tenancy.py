from __future__ import annotations
import uuid
from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import (
    Base,
    Tenant,
    User,
    Patient,
    PredictionJob,
    DEFAULT_TENANT_ID,
    DEFAULT_TENANT_NAME,
    DEFAULT_TENANT_SLUG,
)
from app.db.database import ensure_schema_compatibility


@pytest.fixture
def db_engine():
    """Isolated in-memory SQLite engine for tenancy testing with FK enforcement."""
    from sqlalchemy import event
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    ensure_schema_compatibility(engine)
    return engine


@pytest.fixture
def db_session(db_engine):
    """Provides a transactional session with rollbacks on completion."""
    Session = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_tenant_creation(db_session):
    """Test 1: Tenant creation with valid attributes and default timestamps."""
    tenant = Tenant(
        id=str(uuid.uuid4()),
        name="St. Jude Children's Research Hospital",
        slug="st-jude",
    )
    db_session.add(tenant)
    db_session.commit()
    db_session.refresh(tenant)

    assert tenant.id is not None
    assert tenant.name == "St. Jude Children's Research Hospital"
    assert tenant.slug == "st-jude"
    assert tenant.is_active is True
    assert isinstance(tenant.created_at, datetime)


def test_unique_tenant_slug(db_session):
    """Test 2: Duplicate tenant slug violates uniqueness constraint."""
    tenant1 = Tenant(id=str(uuid.uuid4()), name="Hospital One", slug="hospital-slug")
    db_session.add(tenant1)
    db_session.commit()

    tenant2 = Tenant(id=str(uuid.uuid4()), name="Hospital Two", slug="hospital-slug")
    db_session.add(tenant2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_user_tenant_relationship(db_session):
    """Test 3: User belongs to Tenant with bidirectional relationship."""
    tenant = Tenant(id=str(uuid.uuid4()), name="City Hospital", slug="city-hospital")
    user = User(
        id=str(uuid.uuid4()),
        username="dr_city",
        hashed_password="hashed_test_password",
        role="clinician",
        tenant=tenant,
    )
    db_session.add_all([tenant, user])
    db_session.commit()
    db_session.refresh(tenant)
    db_session.refresh(user)

    assert user.tenant_id == tenant.id
    assert user.tenant.name == "City Hospital"
    assert user in tenant.users


def test_patient_tenant_relationship(db_session):
    """Test 4: Patient belongs to Tenant with bidirectional relationship."""
    tenant = Tenant(id=str(uuid.uuid4()), name="General Clinic", slug="gen-clinic")
    patient = Patient(
        id=str(uuid.uuid4()),
        name="John Doe",
        age=34,
        gender="M",
        weight=70.0,
        height=175.0,
        tenant=tenant,
    )
    db_session.add_all([tenant, patient])
    db_session.commit()
    db_session.refresh(tenant)
    db_session.refresh(patient)

    assert patient.tenant_id == tenant.id
    assert patient.tenant.slug == "gen-clinic"
    assert patient in tenant.patients
    assert patient.is_deleted is False


def test_prediction_job_tenant_relationship(db_session):
    """Test 5: PredictionJob belongs to Tenant with bidirectional relationship."""
    tenant = Tenant(id=str(uuid.uuid4()), name="Neuro Clinic", slug="neuro-clinic")
    job = PredictionJob(
        id=str(uuid.uuid4()),
        status="Validating",
        progress=0,
        tenant=tenant,
    )
    db_session.add_all([tenant, job])
    db_session.commit()
    db_session.refresh(tenant)
    db_session.refresh(job)

    assert job.tenant_id == tenant.id
    assert job.tenant.slug == "neuro-clinic"
    assert job in tenant.prediction_jobs
    assert job.is_deleted is False


def test_patient_creator_relationship(db_session):
    """Test 6: Patient records creator User relationship."""
    tenant = Tenant(id=str(uuid.uuid4()), name="Care Network", slug="care-net")
    clinician = User(
        id=str(uuid.uuid4()),
        username="dr_creator",
        hashed_password="hashed_pw",
        role="clinician",
        tenant=tenant,
    )
    patient = Patient(
        id=str(uuid.uuid4()),
        name="Jane Smith",
        age=28,
        gender="F",
        weight=62.0,
        height=168.0,
        tenant=tenant,
        created_by=clinician,
    )
    db_session.add_all([tenant, clinician, patient])
    db_session.commit()
    db_session.refresh(clinician)
    db_session.refresh(patient)

    assert patient.created_by_user_id == clinician.id
    assert patient.created_by.username == "dr_creator"
    assert patient in clinician.created_patients


def test_prediction_job_creator_relationship(db_session):
    """Test 7: PredictionJob records creator User relationship."""
    tenant = Tenant(id=str(uuid.uuid4()), name="Triage Center", slug="triage-ctr")
    user = User(
        id=str(uuid.uuid4()),
        username="triage_user",
        hashed_password="hashed_pw",
        role="clinician",
        tenant=tenant,
    )
    job = PredictionJob(
        id=str(uuid.uuid4()),
        status="Validating",
        progress=0,
        tenant=tenant,
        created_by=user,
    )
    db_session.add_all([tenant, user, job])
    db_session.commit()
    db_session.refresh(user)
    db_session.refresh(job)

    assert job.created_by_user_id == user.id
    assert job.created_by.username == "triage_user"
    assert job in user.created_prediction_jobs


def test_prediction_job_patient_relationship(db_session):
    """Test 8: PredictionJob links to Patient."""
    tenant = Tenant(id=str(uuid.uuid4()), name="Unified Center", slug="unified-ctr")
    patient = Patient(
        id=str(uuid.uuid4()),
        name="Alex Roe",
        age=45,
        gender="NB",
        weight=75.0,
        height=172.0,
        tenant=tenant,
    )
    job = PredictionJob(
        id=str(uuid.uuid4()),
        status="Validating",
        progress=0,
        tenant=tenant,
        patient=patient,
    )
    db_session.add_all([tenant, patient, job])
    db_session.commit()
    db_session.refresh(patient)
    db_session.refresh(job)

    assert job.patient_id == patient.id
    assert job in patient.jobs
    assert job.patient.name == "Alex Roe"


def test_tenant_consistency_valid(db_session):
    """Test 9: Invariant Tenant A -> Patient A -> Job A is valid and commits."""
    tenant_a = Tenant(id=str(uuid.uuid4()), name="Hospital A", slug="hospital-a")
    user_a = User(
        id=str(uuid.uuid4()),
        username="clinician_a",
        hashed_password="pw",
        role="clinician",
        tenant=tenant_a,
    )
    patient_a = Patient(
        id=str(uuid.uuid4()),
        name="Patient A",
        age=50,
        gender="F",
        weight=65.0,
        height=160.0,
        tenant=tenant_a,
        created_by=user_a,
    )
    job_a = PredictionJob(
        id=str(uuid.uuid4()),
        status="Completed",
        progress=100,
        tenant=tenant_a,
        created_by=user_a,
        patient=patient_a,
    )
    db_session.add_all([tenant_a, user_a, patient_a, job_a])
    db_session.commit()

    assert job_a.tenant_id == patient_a.tenant_id == user_a.tenant_id == tenant_a.id


def test_tenant_consistency_violation_job_patient(db_session):
    """Test 10: Invariant violation: PredictionJob in Tenant B referencing Patient in Tenant A is rejected."""
    tenant_a = Tenant(id=str(uuid.uuid4()), name="Hospital A", slug="hosp-a-viol")
    tenant_b = Tenant(id=str(uuid.uuid4()), name="Hospital B", slug="hosp-b-viol")
    db_session.add_all([tenant_a, tenant_b])
    db_session.commit()

    patient_a = Patient(
        id=str(uuid.uuid4()),
        name="Patient A",
        age=50,
        gender="M",
        weight=80.0,
        height=180.0,
        tenant=tenant_a,
    )
    db_session.add(patient_a)
    db_session.commit()

    # Attempt to attach patient_a (tenant_a) to job in tenant_b
    with pytest.raises(ValueError, match="Tenant consistency violation"):
        job_b = PredictionJob(
            id=str(uuid.uuid4()),
            status="Validating",
            tenant_id=tenant_b.id,
            patient=patient_a,
        )
        db_session.add(job_b)
        db_session.commit()
    db_session.rollback()


def test_tenant_consistency_violation_patient_creator(db_session):
    """Test 11: Invariant violation: Patient in Tenant B referencing creator in Tenant A is rejected."""
    tenant_a = Tenant(id=str(uuid.uuid4()), name="Hospital A", slug="hosp-a-creator")
    tenant_b = Tenant(id=str(uuid.uuid4()), name="Hospital B", slug="hosp-b-creator")
    user_a = User(
        id=str(uuid.uuid4()),
        username="user_a",
        hashed_password="pw",
        role="clinician",
        tenant=tenant_a,
    )
    db_session.add_all([tenant_a, tenant_b, user_a])
    db_session.commit()

    with pytest.raises(ValueError, match="Tenant consistency violation"):
        patient_b = Patient(
            id=str(uuid.uuid4()),
            name="Patient B",
            age=30,
            gender="F",
            weight=60.0,
            height=165.0,
            tenant_id=tenant_b.id,
            created_by=user_a,
        )
        db_session.add(patient_b)
        db_session.commit()
    db_session.rollback()


def test_tenant_consistency_violation_job_creator(db_session):
    """Test 12: Invariant violation: PredictionJob in Tenant B referencing creator in Tenant A is rejected."""
    tenant_a = Tenant(id=str(uuid.uuid4()), name="Hospital A", slug="hosp-a-job-user")
    tenant_b = Tenant(id=str(uuid.uuid4()), name="Hospital B", slug="hosp-b-job-user")
    user_a = User(
        id=str(uuid.uuid4()),
        username="user_a_job",
        hashed_password="pw",
        role="clinician",
        tenant=tenant_a,
    )
    db_session.add_all([tenant_a, tenant_b, user_a])
    db_session.commit()

    with pytest.raises(ValueError, match="Tenant consistency violation"):
        job_b = PredictionJob(
            id=str(uuid.uuid4()),
            status="Validating",
            tenant_id=tenant_b.id,
            created_by=user_a,
        )
        db_session.add(job_b)
        db_session.commit()
    db_session.rollback()


def test_model_validations():
    """Test 13: Model-level validation guards against empty slugs, empty names, and invalid roles."""
    with pytest.raises(ValueError, match="Tenant slug cannot be empty"):
        Tenant(id=str(uuid.uuid4()), name="Name", slug="")

    with pytest.raises(ValueError, match="Tenant name cannot be empty"):
        Tenant(id=str(uuid.uuid4()), name="", slug="slug")

    with pytest.raises(ValueError, match="Invalid user role"):
        User(id=str(uuid.uuid4()), username="u", hashed_password="p", role="superadmin")

    with pytest.raises(ValueError, match="Patient tenant_id cannot be empty"):
        Patient(id=str(uuid.uuid4()), name="P", age=20, gender="F", weight=50, height=160, tenant_id="")

    with pytest.raises(ValueError, match="PredictionJob tenant_id cannot be empty"):
        PredictionJob(id=str(uuid.uuid4()), status="Validating", tenant_id="")


def test_historical_backfill_and_idempotency():
    """Test 14: Historical legacy schema backfills tenant_id cleanly and idempotently."""
    legacy_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create legacy tables without tenant columns
    with legacy_engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE patients ("
                "id VARCHAR PRIMARY KEY, name VARCHAR, age INTEGER, gender VARCHAR, "
                "weight FLOAT, height FLOAT, medical_history VARCHAR, vital_signs JSON, "
                "consent_given BOOLEAN, consent_date TIMESTAMP, data_retention_opt_in BOOLEAN, "
                "created_at TIMESTAMP, status VARCHAR, last_visit TIMESTAMP)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE prediction_jobs ("
                "id VARCHAR PRIMARY KEY, patient_id VARCHAR, status VARCHAR, progress INTEGER, "
                "prediction_label VARCHAR, probability_seizure FLOAT, confidence_band VARCHAR, "
                "shap_explanation JSON, eeg_visualization JSON, detected_dataset VARCHAR, "
                "detection_confidence FLOAT, selected_model VARCHAR, error VARCHAR, "
                "created_at TIMESTAMP, completed_at TIMESTAMP)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE users ("
                "id VARCHAR PRIMARY KEY, username VARCHAR UNIQUE, hashed_password VARCHAR, role VARCHAR)"
            )
        )

        # Seed legacy data
        pat_id = str(uuid.uuid4())
        conn.execute(
            text(
                "INSERT INTO patients (id, name, age, gender, weight, height, status) "
                "VALUES (:id, 'Historical Patient', 60, 'M', 75, 175, 'active')"
            ),
            {"id": pat_id},
        )
        job_id = str(uuid.uuid4())
        conn.execute(
            text(
                "INSERT INTO prediction_jobs (id, patient_id, status, progress, prediction_label, probability_seizure) "
                "VALUES (:id, :pid, 'Completed', 100, 'seizure', 0.95)"
            ),
            {"id": job_id, "pid": pat_id},
        )
        user_id = str(uuid.uuid4())
        conn.execute(
            text(
                "INSERT INTO users (id, username, hashed_password, role) "
                "VALUES (:id, 'legacy_clinician', 'hashed_pw', 'clinician')"
            ),
            {"id": user_id},
        )

    # Run migration pass 1
    ensure_schema_compatibility(legacy_engine)

    inspector = inspect(legacy_engine)
    assert "tenants" in inspector.get_table_names()
    pat_cols = {col["name"] for col in inspector.get_columns("patients")}
    assert "tenant_id" in pat_cols
    assert "created_by_user_id" in pat_cols
    assert "is_deleted" in pat_cols

    job_cols = {col["name"] for col in inspector.get_columns("prediction_jobs")}
    assert "tenant_id" in job_cols
    assert "created_by_user_id" in job_cols
    assert "is_deleted" in job_cols
    assert "worker_id" in job_cols
    assert "heartbeat_at" in job_cols
    assert "lease_expires_at" in job_cols

    # Verify backfilled values
    with legacy_engine.begin() as conn:
        pat_res = conn.execute(text("SELECT tenant_id, created_by_user_id, is_deleted FROM patients")).fetchone()
        assert pat_res[0] == DEFAULT_TENANT_ID
        assert pat_res[1] is None  # Unknown historical creator preserved as NULL
        assert pat_res[2] in (False, 0)

        job_res = conn.execute(text("SELECT tenant_id, created_by_user_id, is_deleted, probability_seizure FROM prediction_jobs")).fetchone()
        assert job_res[0] == DEFAULT_TENANT_ID
        assert job_res[1] is None  # Unknown historical creator preserved as NULL
        assert job_res[2] in (False, 0)
        assert job_res[3] == 0.95  # Results completely preserved

        user_res = conn.execute(text("SELECT tenant_id, is_active, token_version FROM users")).fetchone()
        assert user_res[0] == DEFAULT_TENANT_ID
        assert user_res[1] in (True, 1)
        assert user_res[2] == 1

        tenant_res = conn.execute(text("SELECT id, name, slug FROM tenants")).fetchall()
        assert len(tenant_res) == 1
        assert tenant_res[0][0] == DEFAULT_TENANT_ID
        assert tenant_res[0][1] == DEFAULT_TENANT_NAME
        assert tenant_res[0][2] == DEFAULT_TENANT_SLUG

    # Run migration pass 2 (idempotency test)
    ensure_schema_compatibility(legacy_engine)

    # Verify counts didn't change
    with legacy_engine.begin() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM tenants")).scalar() == 1
        assert conn.execute(text("SELECT COUNT(*) FROM patients")).scalar() == 1
        assert conn.execute(text("SELECT COUNT(*) FROM prediction_jobs")).scalar() == 1
        assert conn.execute(text("SELECT COUNT(*) FROM users")).scalar() == 1


def test_existing_prediction_job_lifecycle_fields_preserved(db_session):
    """Test 15: Worker lifecycle leases and ML explanations persist alongside tenancy."""
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    job = PredictionJob(
        id=job_id,
        status="Validating",
        progress=50,
        worker_id="worker-node-1:5000",
        heartbeat_at=now,
        lease_expires_at=now,
        shap_explanation={"base_value": 0.5, "features": []},
        eeg_visualization={"dataset": "chbmit", "channels": []},
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    assert job.tenant_id == DEFAULT_TENANT_ID
    assert job.worker_id == "worker-node-1:5000"
    assert job.shap_explanation == {"base_value": 0.5, "features": []}
    assert job.eeg_visualization == {"dataset": "chbmit", "channels": []}


def test_tenant_deletion_safety_blocks_cascading(db_session, db_engine):
    """Test 16 (Scenario A): Tenant deletion is safely blocked and NEVER silently deletes clinical records."""
    tenant_id = str(uuid.uuid4())
    tenant = Tenant(id=tenant_id, name="Hospital Block Delete", slug="hosp-block-del")
    user = User(id=str(uuid.uuid4()), username="doc_block", hashed_password="pw", role="clinician", tenant=tenant)
    patient = Patient(id=str(uuid.uuid4()), name="Protected Patient", age=45, tenant=tenant, created_by=user)
    job = PredictionJob(id=str(uuid.uuid4()), status="Completed", tenant=tenant, patient=patient, created_by=user)
    db_session.add_all([tenant, user, patient, job])
    db_session.commit()

    # 1. ORM deletion attempt fails safely without cascading
    with pytest.raises((ValueError, IntegrityError)):
        db_session.delete(tenant)
        db_session.commit()
    db_session.rollback()

    # 2. Raw SQL deletion attempt is strictly blocked by Foreign Key RESTRICT
    with db_engine.begin() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(text("DELETE FROM tenants WHERE id = :tid"), {"tid": tenant_id})

    # Verify that zero clinical records or users disappeared
    p_check = db_session.query(Patient).filter_by(id=patient.id).first()
    j_check = db_session.query(PredictionJob).filter_by(id=job.id).first()
    u_check = db_session.query(User).filter_by(id=user.id).first()
    t_check = db_session.query(Tenant).filter_by(id=tenant_id).first()

    assert t_check is not None
    assert p_check is not None
    assert j_check is not None
    assert u_check is not None
    assert j_check.patient_id == patient.id


def test_patient_deletion_preserves_prediction_history(db_session):
    """Test 17 (Scenario B): Hard deletion of a patient does not unexpectedly destroy prediction history."""
    tenant = Tenant(id=str(uuid.uuid4()), name="Clinical Preservation Clinic", slug="preserv-clinic")
    patient = Patient(id=str(uuid.uuid4()), name="Preserved Patient", age=50, tenant=tenant)
    job = PredictionJob(
        id=str(uuid.uuid4()),
        status="Completed",
        tenant=tenant,
        patient=patient,
        shap_explanation={"base_value": 0.4},
        eeg_visualization={"dataset": "chbmit"},
    )
    db_session.add_all([tenant, patient, job])
    db_session.commit()

    # Deleting patient disassociates the job (patient_id becomes NULL) rather than deleting it
    db_session.delete(patient)
    db_session.commit()

    j_after = db_session.query(PredictionJob).filter_by(id=job.id).first()
    assert j_after is not None
    assert j_after.patient_id is None
    assert j_after.shap_explanation == {"base_value": 0.4}
    assert j_after.eeg_visualization == {"dataset": "chbmit"}


def test_user_deletion_preserves_patient_and_job(db_session):
    """Test 18 (Scenario C): Deleting a creator user nullifies created_by_user_id and preserves records."""
    tenant = Tenant(id=str(uuid.uuid4()), name="Staff Department", slug="staff-dept")
    clinician = User(id=str(uuid.uuid4()), username="dr_leaving", hashed_password="pw", role="clinician", tenant=tenant)
    patient = Patient(id=str(uuid.uuid4()), name="Active Patient", age=32, tenant=tenant, created_by=clinician)
    job = PredictionJob(id=str(uuid.uuid4()), status="Completed", tenant=tenant, created_by=clinician, patient=patient)
    db_session.add_all([tenant, clinician, patient, job])
    db_session.commit()

    # Delete clinician user
    db_session.delete(clinician)
    db_session.commit()

    # Clinical records remain intact with creator set to NULL
    p_after = db_session.query(Patient).filter_by(id=patient.id).first()
    j_after = db_session.query(PredictionJob).filter_by(id=job.id).first()
    t_after = db_session.query(Tenant).filter_by(id=tenant.id).first()

    assert p_after is not None
    assert p_after.created_by_user_id is None
    assert j_after is not None
    assert j_after.created_by_user_id is None
    assert t_after is not None


def test_tenant_deactivation_preserves_all_records(db_session):
    """Test 19 (Scenario D): Deactivating a tenant preserves all users, patients, jobs, and ML payloads."""
    tenant = Tenant(id=str(uuid.uuid4()), name="Deactivating Hospital", slug="deactivating-hosp", is_active=True)
    user = User(id=str(uuid.uuid4()), username="active_user", hashed_password="pw", role="clinician", tenant=tenant)
    patient = Patient(id=str(uuid.uuid4()), name="Patient Record", age=62, tenant=tenant, created_by=user)
    now = datetime.now(timezone.utc)
    job = PredictionJob(
        id=str(uuid.uuid4()),
        status="Completed",
        tenant=tenant,
        patient=patient,
        created_by=user,
        worker_id="worker-01",
        heartbeat_at=now,
        lease_expires_at=now,
        shap_explanation={"importance": [0.1, 0.2]},
        eeg_visualization={"channels": ["C3-P3"]},
    )
    db_session.add_all([tenant, user, patient, job])
    db_session.commit()

    # Deactivate tenant
    tenant.is_active = False
    db_session.commit()

    # Verify everything remains intact
    t_check = db_session.query(Tenant).filter_by(id=tenant.id).first()
    u_check = db_session.query(User).filter_by(id=user.id).first()
    p_check = db_session.query(Patient).filter_by(id=patient.id).first()
    j_check = db_session.query(PredictionJob).filter_by(id=job.id).first()

    assert t_check.is_active is False
    assert u_check is not None
    assert p_check is not None
    assert j_check is not None
    assert j_check.worker_id == "worker-01"
    assert j_check.shap_explanation == {"importance": [0.1, 0.2]}
    assert j_check.eeg_visualization == {"channels": ["C3-P3"]}


def test_sqlite_foreign_key_enforcement_active(db_engine, db_session):
    """Test 20 (Phase 3): PRAGMA foreign_keys is active and rejects orphaned/invalid foreign keys."""
    with db_engine.connect() as conn:
        pragma_val = conn.execute(text("PRAGMA foreign_keys")).scalar()
        assert pragma_val == 1

    # Attempt inserting patient referencing non-existent tenant
    with pytest.raises(IntegrityError):
        invalid_patient = Patient(
            id=str(uuid.uuid4()),
            name="Invalid Tenant Patient",
            age=30,
            tenant_id="00000000-0000-0000-0000-999999999999",
        )
        db_session.add(invalid_patient)
        db_session.commit()
    db_session.rollback()


def test_migration_failure_safety_disposable_db():
    """Test 21 (Phase 6): Migration repeated runs are idempotent and failure modes do not corrupt data."""
    disposable_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with disposable_engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE patients (id VARCHAR PRIMARY KEY, name VARCHAR, age INTEGER)")
        )
        conn.execute(
            text("INSERT INTO patients (id, name, age) VALUES ('p-disp-1', 'Disposable Patient', 45)")
        )

    # 1. First run creates tables & backfills
    ensure_schema_compatibility(disposable_engine)
    with disposable_engine.begin() as conn:
        assert conn.execute(text("SELECT tenant_id FROM patients WHERE id = 'p-disp-1'")).scalar() == DEFAULT_TENANT_ID

    # 2. Second run is strictly idempotent
    ensure_schema_compatibility(disposable_engine)
    with disposable_engine.begin() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM tenants")).scalar() == 1
        assert conn.execute(text("SELECT COUNT(*) FROM patients")).scalar() == 1

    # 3. Third run produces identical invariant state
    ensure_schema_compatibility(disposable_engine)
    with disposable_engine.begin() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM tenants")).scalar() == 1
        assert conn.execute(text("SELECT tenant_id FROM patients WHERE id = 'p-disp-1'")).scalar() == DEFAULT_TENANT_ID

