"""Snapshot CRUD routes — create, list, restore, delete CRIU+GPU snapshots.

These endpoints are called by the ai-gateway's SnapgpuClient over HTTP.
The SnapshotManager does the heavy lifting (CRIU dump/restore + cuda-checkpoint).
"""

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..builder.snapshot import SnapshotManager
from ..db import get_session, SnapshotModel
from sqlmodel import select

router = APIRouter(prefix="/v1/snapshots", tags=["snapshots"])

# Shared SnapshotManager instance (one per gateway process)
_manager = SnapshotManager()


# ── Request/Response models ──────────────────────────────────────────────────

class CreateSnapshotRequest(BaseModel):
    app_name: str
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    include_gpu: bool = True
    """PID of the process to snapshot. If not set, auto-detected from the
    warm container pool for the given app_name."""
    pid: Optional[int] = None
    container_id: Optional[str] = None


class CreateSnapshotResponse(BaseModel):
    snapshot_id: Optional[str] = None
    size_bytes: int = 0
    gpu_memory_included: bool = False
    error: Optional[str] = None


class SnapshotInfo(BaseModel):
    snapshot_id: str
    app_name: str
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    image_tag: str = ""
    size_bytes: int = 0
    gpu_memory_included: bool = False
    created_at: str = ""


class SnapshotListResponse(BaseModel):
    snapshots: list[SnapshotInfo]


class RestoreResponse(BaseModel):
    success: bool
    restored_pid: Optional[int] = None
    error: Optional[str] = None


# ── Routes ───────────────────────────────────────────────────────────────────

@router.post("", response_model=CreateSnapshotResponse)
async def create_snapshot(body: CreateSnapshotRequest):
    """Create a CRIU snapshot of a running process (optionally with GPU memory).

    The snapshot captures the full process state so subsequent boots can restore
    in ~2-5 seconds instead of cold-loading models for ~2 minutes.

    Requires CRIU installed and (for GPU snapshots) cuda-checkpoint + driver 570+.
    Returns { snapshot_id: null } when CRIU is unavailable (graceful degradation).
    """
    if not _manager.criu_available:
        return CreateSnapshotResponse(
            error="CRIU not available on this host — snapshot skipped",
        )

    pid = body.pid
    container_id = body.container_id or ""

    if not pid:
        # Auto-detect PID from warm container pool
        # For now, require explicit PID. Future: query ContainerPool.
        return CreateSnapshotResponse(
            error="pid is required (auto-detection from container pool not yet implemented)",
        )

    snapshot_id = _manager.create(
        container_id=container_id,
        app_name=body.app_name,
        pid=pid,
        function_name=body.function_name,
        class_name=body.class_name,
        include_gpu=body.include_gpu,
    )

    if not snapshot_id:
        return CreateSnapshotResponse(error="Snapshot creation failed — check gateway logs")

    # Read back the snapshot metadata from DB
    with get_session() as session:
        snap = session.exec(
            select(SnapshotModel).where(SnapshotModel.snapshot_id == snapshot_id)
        ).first()

    return CreateSnapshotResponse(
        snapshot_id=snapshot_id,
        size_bytes=snap.size_bytes if snap else 0,
        gpu_memory_included=snap.gpu_memory_included if snap else False,
    )


@router.get("", response_model=SnapshotListResponse)
async def list_snapshots():
    """List all snapshots stored on this gateway."""
    with get_session() as session:
        snaps = session.exec(select(SnapshotModel)).all()
        return SnapshotListResponse(
            snapshots=[
                SnapshotInfo(
                    snapshot_id=s.snapshot_id,
                    app_name=s.app_name,
                    function_name=s.function_name,
                    class_name=s.class_name,
                    image_tag=s.image_tag or "",
                    size_bytes=s.size_bytes or 0,
                    gpu_memory_included=s.gpu_memory_included or False,
                    created_at=str(s.created_at) if hasattr(s, 'created_at') else "",
                )
                for s in snaps
            ]
        )


@router.post("/{snapshot_id}/restore", response_model=RestoreResponse)
async def restore_snapshot(snapshot_id: str):
    """Restore a process from a previously created snapshot.

    The CRIU restore recreates the process tree with all file descriptors,
    memory pages, and (if captured) GPU memory state intact. Typical restore
    time is 2-5 seconds for a ~8GB model.
    """
    pid = _manager.restore(snapshot_id)
    if pid is None:
        return RestoreResponse(success=False, error=f"Failed to restore snapshot {snapshot_id}")
    return RestoreResponse(success=True, restored_pid=pid)


@router.delete("/{snapshot_id}")
async def delete_snapshot(snapshot_id: str):
    """Delete a snapshot and free disk space."""
    _manager.delete(snapshot_id)
    return {"deleted": True, "snapshot_id": snapshot_id}
