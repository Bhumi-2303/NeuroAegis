from __future__ import annotations
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from sqlalchemy.pool import StaticPool

from app.core.config import settings

engine_kwargs = {}
if settings.DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
    if ":memory:" in settings.DATABASE_URL:
        engine_kwargs["poolclass"] = StaticPool

engine = create_engine(settings.DATABASE_URL, **engine_kwargs)

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
    Ensure existing tables have newly added columns in a safe, idempotent manner.
    Works for both PostgreSQL and SQLite without requiring external migration tooling.
    Called explicitly by API lifespan or migration runners; workers do NOT run migrations.
    """
    import logging
    from sqlalchemy import inspect, text

    target_engine = engine_instance or engine
    logger = logging.getLogger("neuroaegis.database")

    try:
        inspector = inspect(target_engine)
        tables = inspector.get_table_names()
        if "prediction_jobs" not in tables:
            return

        existing_columns = {col["name"] for col in inspector.get_columns("prediction_jobs")}
        with target_engine.begin() as conn:
            if "worker_id" not in existing_columns:
                logger.info("Migrating schema: adding 'worker_id' column to prediction_jobs")
                conn.execute(text("ALTER TABLE prediction_jobs ADD COLUMN worker_id VARCHAR"))
            if "heartbeat_at" not in existing_columns:
                logger.info("Migrating schema: adding 'heartbeat_at' column to prediction_jobs")
                conn.execute(text("ALTER TABLE prediction_jobs ADD COLUMN heartbeat_at TIMESTAMP"))
            if "lease_expires_at" not in existing_columns:
                logger.info("Migrating schema: adding 'lease_expires_at' column to prediction_jobs")
                conn.execute(text("ALTER TABLE prediction_jobs ADD COLUMN lease_expires_at TIMESTAMP"))
    except Exception as exc:
        logger.warning(f"Notice during schema compatibility verification: {exc}")
