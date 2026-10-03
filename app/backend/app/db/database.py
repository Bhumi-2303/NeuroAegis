from __future__ import annotations
from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

from sqlalchemy.pool import StaticPool

from app.core.config import settings

engine_kwargs = {}
if settings.DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
    if ":memory:" in settings.DATABASE_URL:
        engine_kwargs["poolclass"] = StaticPool

engine = create_engine(settings.DATABASE_URL, **engine_kwargs)

if settings.DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_schema_compatibility(engine_instance=None) -> None:
    """
    Ensure existing tables have newly added columns, default tenant, and indexes in a safe, idempotent manner.
    Works for both PostgreSQL and SQLite without requiring external migration tooling.
    Called explicitly by API lifespan or migration runners; workers do NOT run migrations.
    """
    import logging
    from datetime import datetime, timezone
    from sqlalchemy import inspect, text

    target_engine = engine_instance or engine
    logger = logging.getLogger("neuroaegis.database")

    try:
        from app.db.models import (
            Base,
            DEFAULT_TENANT_ID,
            DEFAULT_TENANT_NAME,
            DEFAULT_TENANT_SLUG,
        )

        with target_engine.begin() as conn:
            inspector = inspect(conn)
            tables = set(inspector.get_table_names())

            # 1. Ensure 'tenants' table exists
            if "tenants" not in tables:
                logger.info("Migrating schema: creating 'tenants' table")
                Base.metadata.tables["tenants"].create(bind=conn, checkfirst=True)
                tables.add("tenants")

            # 2. Ensure default organization tenant exists
            res = conn.execute(
                text("SELECT id FROM tenants WHERE id = :id OR slug = :slug"),
                {"id": DEFAULT_TENANT_ID, "slug": DEFAULT_TENANT_SLUG},
            ).fetchone()
            if not res:
                logger.info("Migrating schema: inserting default organization tenant")
                conn.execute(
                    text(
                        "INSERT INTO tenants (id, name, slug, is_active, created_at) "
                        "VALUES (:id, :name, :slug, :is_active, :created_at)"
                    ),
                    {
                        "id": DEFAULT_TENANT_ID,
                        "name": DEFAULT_TENANT_NAME,
                        "slug": DEFAULT_TENANT_SLUG,
                        "is_active": True,
                        "created_at": datetime.now(timezone.utc),
                    },
                )

            # 3. Migrate 'users' table if present
            if "users" in tables:
                user_cols = {col["name"] for col in inspector.get_columns("users")}
                if "tenant_id" not in user_cols:
                    logger.info("Migrating schema: adding 'tenant_id' column to users")
                    conn.execute(text("ALTER TABLE users ADD COLUMN tenant_id VARCHAR"))
                if "is_active" not in user_cols:
                    logger.info("Migrating schema: adding 'is_active' column to users")
                    conn.execute(text("ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT TRUE"))
                if "token_version" not in user_cols:
                    logger.info("Migrating schema: adding 'token_version' column to users")
                    conn.execute(text("ALTER TABLE users ADD COLUMN token_version INTEGER DEFAULT 1"))
                if "created_at" not in user_cols:
                    logger.info("Migrating schema: adding 'created_at' column to users")
                    conn.execute(text("ALTER TABLE users ADD COLUMN created_at TIMESTAMP"))

                # Backfill users
                conn.execute(
                    text("UPDATE users SET tenant_id = :def_t WHERE tenant_id IS NULL"),
                    {"def_t": DEFAULT_TENANT_ID},
                )
                conn.execute(text("UPDATE users SET is_active = TRUE WHERE is_active IS NULL"))
                conn.execute(text("UPDATE users SET token_version = 1 WHERE token_version IS NULL"))

            # 4. Migrate 'patients' table if present
            if "patients" in tables:
                patient_cols = {col["name"] for col in inspector.get_columns("patients")}
                if "tenant_id" not in patient_cols:
                    logger.info("Migrating schema: adding 'tenant_id' column to patients")
                    conn.execute(text("ALTER TABLE patients ADD COLUMN tenant_id VARCHAR"))
                if "created_by_user_id" not in patient_cols:
                    logger.info("Migrating schema: adding 'created_by_user_id' column to patients")
                    conn.execute(text("ALTER TABLE patients ADD COLUMN created_by_user_id VARCHAR"))
                if "is_deleted" not in patient_cols:
                    logger.info("Migrating schema: adding 'is_deleted' column to patients")
                    conn.execute(text("ALTER TABLE patients ADD COLUMN is_deleted BOOLEAN DEFAULT FALSE"))

                # Backfill patients
                conn.execute(
                    text("UPDATE patients SET tenant_id = :def_t WHERE tenant_id IS NULL"),
                    {"def_t": DEFAULT_TENANT_ID},
                )
                conn.execute(text("UPDATE patients SET is_deleted = FALSE WHERE is_deleted IS NULL"))
                # Note: created_by_user_id remains NULL for historical patients where creator is unknown

            # 5. Migrate 'prediction_jobs' table if present
            if "prediction_jobs" in tables:
                job_cols = {col["name"] for col in inspector.get_columns("prediction_jobs")}
                if "worker_id" not in job_cols:
                    logger.info("Migrating schema: adding 'worker_id' column to prediction_jobs")
                    conn.execute(text("ALTER TABLE prediction_jobs ADD COLUMN worker_id VARCHAR"))
                if "heartbeat_at" not in job_cols:
                    logger.info("Migrating schema: adding 'heartbeat_at' column to prediction_jobs")
                    conn.execute(text("ALTER TABLE prediction_jobs ADD COLUMN heartbeat_at TIMESTAMP"))
                if "lease_expires_at" not in job_cols:
                    logger.info("Migrating schema: adding 'lease_expires_at' column to prediction_jobs")
                    conn.execute(text("ALTER TABLE prediction_jobs ADD COLUMN lease_expires_at TIMESTAMP"))
                if "tenant_id" not in job_cols:
                    logger.info("Migrating schema: adding 'tenant_id' column to prediction_jobs")
                    conn.execute(text("ALTER TABLE prediction_jobs ADD COLUMN tenant_id VARCHAR"))
                if "created_by_user_id" not in job_cols:
                    logger.info("Migrating schema: adding 'created_by_user_id' column to prediction_jobs")
                    conn.execute(text("ALTER TABLE prediction_jobs ADD COLUMN created_by_user_id VARCHAR"))
                if "request_id" not in job_cols:
                    logger.info("Migrating schema: adding 'request_id' column to prediction_jobs")
                    conn.execute(text("ALTER TABLE prediction_jobs ADD COLUMN request_id VARCHAR(64)"))
                if "is_deleted" not in job_cols:
                    logger.info("Migrating schema: adding 'is_deleted' column to prediction_jobs")
                    conn.execute(text("ALTER TABLE prediction_jobs ADD COLUMN is_deleted BOOLEAN DEFAULT FALSE"))

                # Backfill prediction_jobs
                conn.execute(
                    text("UPDATE prediction_jobs SET tenant_id = :def_t WHERE tenant_id IS NULL"),
                    {"def_t": DEFAULT_TENANT_ID},
                )
                conn.execute(text("UPDATE prediction_jobs SET is_deleted = FALSE WHERE is_deleted IS NULL"))
                # Note: created_by_user_id remains NULL for historical jobs where creator is unknown

            # 6. Ensure 'refresh_tokens' table exists (Prompt 9.2)
            if "refresh_tokens" not in tables:
                logger.info("Migrating schema: creating 'refresh_tokens' table")
                Base.metadata.tables["refresh_tokens"].create(bind=conn, checkfirst=True)
                tables.add("refresh_tokens")

            # 7. Ensure 'audit_events' table exists (Phase 10.4)
            if "audit_events" not in tables:
                logger.info("Migrating schema: creating 'audit_events' table")
                Base.metadata.tables["audit_events"].create(bind=conn, checkfirst=True)
                tables.add("audit_events")

            # 8. Ensure indexes exist
            indexes = [
                ("ix_tenants_slug", "tenants", ["slug"]),
                ("ix_users_tenant_id", "users", ["tenant_id"]),
                ("ix_patients_tenant_id", "patients", ["tenant_id"]),
                ("ix_patients_created_by_user_id", "patients", ["created_by_user_id"]),
                ("ix_patients_tenant_id_composite", "patients", ["tenant_id", "id"]),
                ("ix_prediction_jobs_tenant_id", "prediction_jobs", ["tenant_id"]),
                ("ix_prediction_jobs_created_by_user_id", "prediction_jobs", ["created_by_user_id"]),
                ("ix_prediction_jobs_patient_id", "prediction_jobs", ["patient_id"]),
                ("ix_prediction_jobs_tenant_patient", "prediction_jobs", ["tenant_id", "patient_id"]),
                ("ix_prediction_jobs_request_id", "prediction_jobs", ["request_id"]),
                ("ix_refresh_tokens_token_hash", "refresh_tokens", ["token_hash"]),
                ("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"]),
                ("ix_refresh_tokens_tenant_id", "refresh_tokens", ["tenant_id"]),
                ("ix_refresh_tokens_expires_at", "refresh_tokens", ["expires_at"]),
                ("ix_audit_events_tenant_occurred", "audit_events", ["tenant_id", "occurred_at"]),
                ("ix_audit_events_tenant_event_type", "audit_events", ["tenant_id", "event_type"]),
                ("ix_audit_events_tenant_patient", "audit_events", ["tenant_id", "patient_id"]),
                ("ix_audit_events_tenant_job", "audit_events", ["tenant_id", "job_id"]),
                ("ix_audit_events_request_id", "audit_events", ["request_id"]),
                ("ix_audit_events_actor", "audit_events", ["actor_id"]),
            ]
            for idx_name, table_name, cols in indexes:
                if table_name in tables:
                    cols_sql = ", ".join(cols)
                    try:
                        conn.execute(text(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name} ({cols_sql})"))
                    except Exception as idx_exc:
                        logger.debug(f"Index creation notice ({idx_name}): {idx_exc}")

            # 9. Append-only triggers for audit_events (Phase 10.4)
            if "audit_events" in tables:
                if conn.dialect.name == "sqlite":
                    try:
                        conn.execute(text("""
                            CREATE TRIGGER IF NOT EXISTS trg_audit_no_update
                            BEFORE UPDATE ON audit_events
                            BEGIN
                                SELECT RAISE(ABORT, 'audit_events table is append-only: updates are forbidden');
                            END;
                        """))
                        conn.execute(text("""
                            CREATE TRIGGER IF NOT EXISTS trg_audit_no_delete
                            BEFORE DELETE ON audit_events
                            BEGIN
                                SELECT RAISE(ABORT, 'audit_events table is append-only: deletions are forbidden');
                            END;
                        """))
                    except Exception as trg_exc:
                        logger.debug(f"SQLite trigger notice: {trg_exc}")
                elif conn.dialect.name == "postgresql":
                    try:
                        conn.execute(text("""
                            CREATE OR REPLACE FUNCTION prevent_audit_tampering()
                            RETURNS TRIGGER AS $$
                            BEGIN
                                RAISE EXCEPTION 'audit_events table is append-only: updates and deletions are forbidden';
                            END;
                            $$ LANGUAGE plpgsql;
                        """))
                        conn.execute(text("""
                            DO $$
                            BEGIN
                                IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_audit_events_immutable') THEN
                                    CREATE TRIGGER trg_audit_events_immutable
                                    BEFORE UPDATE OR DELETE ON audit_events
                                    FOR EACH ROW EXECUTE FUNCTION prevent_audit_tampering();
                                END IF;
                            END $$;
                        """))
                    except Exception as trg_exc:
                        logger.debug(f"PostgreSQL trigger notice: {trg_exc}")

            # 10. PostgreSQL foreign key constraints (idempotent)
            if conn.dialect.name == "postgresql":
                fk_constraints = [
                    ("fk_users_tenant", "users", "FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE RESTRICT"),
                    ("fk_patients_tenant", "patients", "FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE RESTRICT"),
                    ("fk_patients_created_by", "patients", "FOREIGN KEY (created_by_user_id) REFERENCES users(id) ON DELETE SET NULL"),
                    ("fk_prediction_jobs_tenant", "prediction_jobs", "FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE RESTRICT"),
                    ("fk_prediction_jobs_created_by", "prediction_jobs", "FOREIGN KEY (created_by_user_id) REFERENCES users(id) ON DELETE SET NULL"),
                    ("fk_refresh_tokens_user", "refresh_tokens", "FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE"),
                    ("fk_refresh_tokens_tenant", "refresh_tokens", "FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE RESTRICT"),
                    ("fk_audit_events_tenant", "audit_events", "FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE RESTRICT"),
                ]
                for fk_name, tbl_name, fk_clause in fk_constraints:
                    if tbl_name in tables:
                        try:
                            exists = conn.execute(
                                text("SELECT 1 FROM pg_constraint WHERE conname = :cname"),
                                {"cname": fk_name},
                            ).scalar()
                            if not exists:
                                conn.execute(
                                    text(f"ALTER TABLE {tbl_name} ADD CONSTRAINT {fk_name} {fk_clause}")
                                )
                        except Exception as fk_exc:
                            logger.debug(f"FK constraint notice ({fk_name}): {fk_exc}")

            # 9. Tenancy integrity verification
            if "prediction_jobs" in tables and "patients" in tables:
                inconsistent = conn.execute(
                    text(
                        "SELECT COUNT(*) FROM prediction_jobs j "
                        "JOIN patients p ON j.patient_id = p.id "
                        "WHERE j.tenant_id != p.tenant_id"
                    )
                ).scalar()
                if inconsistent and inconsistent > 0:
                    logger.error(f"Tenancy integrity violation: {inconsistent} jobs have tenant_id != patient.tenant_id")
                else:
                    logger.info("Database schema and tenancy integrity verified successfully.")
    except Exception as exc:
        logger.warning(f"Notice during schema compatibility verification: {exc}")
