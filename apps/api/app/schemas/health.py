from __future__ import annotations
from typing import Any

from pydantic import BaseModel


class HealthCheckSchema(BaseModel):
    status: str
    model_loaded: bool
    version: str
    model_version: str | None = None
    dataset_name: str | None = None
    last_load_time: str | None = None
    details: dict[str, Any] | None = None
