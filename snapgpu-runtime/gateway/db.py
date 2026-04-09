"""Database models for the SnapGPU gateway (SQLite + SQLModel)."""

from __future__ import annotations
import os
import json
from datetime import datetime, timezone
from typing import Optional
from enum import Enum

try:
    from sqlmodel import SQLModel, Field, create_engine, Session, select
except ImportError:
    raise ImportError("sqlmodel is required: pip install sqlmodel")


# ── Enums ─────────────────────────────────────────────────────────────────

class AppStatus(str, Enum):
    ACTIVE = "active"
    DEPLOYING = "deploying"
    STOPPED = "stopped"
    ERROR = "error"


class ContainerStatus(str, Enum):
    PENDING = "pending"
    CREATING = "creating"
    RUNNING = "running"
    SNAPSHOTTING = "snapshotting"
    IDLE = "idle"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Models ────────────────────────────────────────────────────────────────

class AppModel(SQLModel, table=True):
    __tablename__ = "apps"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    status: str = Field(default=AppStatus.ACTIVE)
    spec_json: str = Field(default="{}")  # Full app spec (functions, classes, images)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def spec(self) -> dict:
        return json.loads(self.spec_json)

    @spec.setter
    def spec(self, value: dict):
        self.spec_json = json.dumps(value)


class ContainerModel(SQLModel, table=True):
    __tablename__ = "containers"

    id: Optional[int] = Field(default=None, primary_key=True)
    container_id: str = Field(index=True)  # Docker container ID
    app_name: str = Field(index=True)
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    status: str = Field(default=ContainerStatus.PENDING)
    gpu_device: Optional[str] = None  # e.g., "0", "1"
    gpu_type: Optional[str] = None
    image_tag: str = ""
    snapshot_id: Optional[str] = None  # Reference to snapshot for fast restore
    last_request_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TaskModel(SQLModel, table=True):
    __tablename__ = "tasks"

    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: str = Field(unique=True, index=True)
    app_name: str = Field(index=True)
    function_name: str
    status: str = Field(default=TaskStatus.PENDING)
    result_data: Optional[str] = None  # Base64-encoded pickled result
    error: Optional[str] = None
    container_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


class SnapshotModel(SQLModel, table=True):
    __tablename__ = "snapshots"

    id: Optional[int] = Field(default=None, primary_key=True)
    snapshot_id: str = Field(unique=True, index=True)
    app_name: str = Field(index=True)
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    image_tag: str = ""
    snapshot_path: str = ""  # Path to CRIU snapshot directory
    gpu_memory_included: bool = False
    size_bytes: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ImageModel(SQLModel, table=True):
    __tablename__ = "images"

    id: Optional[int] = Field(default=None, primary_key=True)
    tag: str = Field(unique=True, index=True)
    app_name: str = Field(index=True)
    dockerfile_hash: str = ""  # Content hash for dedup
    size_bytes: int = 0
    built_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Database connection ───────────────────────────────────────────────────

_DB_URL = os.environ.get("SNAPGPU_DB_URL", "sqlite:///snapgpu.db")
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(_DB_URL, echo=False)
        SQLModel.metadata.create_all(_engine)
    return _engine


def get_session() -> Session:
    return Session(get_engine())
