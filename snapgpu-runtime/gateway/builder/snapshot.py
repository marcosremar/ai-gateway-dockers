"""CRIU-based container snapshots for fast cold starts.

Creates a snapshot after the model is loaded (@enter(snap=True)),
so subsequent starts restore from snapshot instead of re-loading.

Requires CRIU and optionally cuda-checkpoint for GPU memory snapshots.
"""

from __future__ import annotations
import os
import subprocess
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from ..db import get_session, SnapshotModel


class SnapshotManager:
    """Creates and restores CRIU snapshots of GPU containers."""

    def __init__(self, snapshot_dir: str = "/var/snapgpu/snapshots"):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._criu_available: Optional[bool] = None
        self._cuda_checkpoint_available: Optional[bool] = None

    @property
    def criu_available(self) -> bool:
        if self._criu_available is None:
            try:
                subprocess.run(["criu", "--version"], capture_output=True, timeout=5)
                self._criu_available = True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self._criu_available = False
        return self._criu_available

    @property
    def cuda_checkpoint_available(self) -> bool:
        if self._cuda_checkpoint_available is None:
            try:
                subprocess.run(["cuda-checkpoint", "--help"], capture_output=True, timeout=5)
                self._cuda_checkpoint_available = True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self._cuda_checkpoint_available = False
        return self._cuda_checkpoint_available

    def create(
        self,
        container_id: str,
        app_name: str,
        pid: int,
        *,
        function_name: Optional[str] = None,
        class_name: Optional[str] = None,
        image_tag: str = "",
        include_gpu: bool = True,
    ) -> Optional[str]:
        """Create a CRIU snapshot of a running container process.

        Args:
            container_id: Docker container ID
            pid: PID of the main process inside the container
            include_gpu: If True, also snapshot GPU memory via cuda-checkpoint

        Returns:
            snapshot_id if successful, None if CRIU not available
        """
        if not self.criu_available:
            print("[snapshot] CRIU not available — skipping snapshot")
            return None

        snapshot_id = f"snap-{uuid.uuid4().hex[:12]}"
        snapshot_path = self.snapshot_dir / snapshot_id
        snapshot_path.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Checkpoint GPU memory (if available and requested)
            gpu_included = False
            if include_gpu and self.cuda_checkpoint_available:
                print(f"[snapshot] Checkpointing GPU memory for PID {pid}...")
                result = subprocess.run(
                    ["cuda-checkpoint", "--toggle", "--pid", str(pid)],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    gpu_included = True
                else:
                    print(f"[snapshot] cuda-checkpoint failed: {result.stderr}")

            # Step 2: CRIU checkpoint
            print(f"[snapshot] Creating CRIU checkpoint for PID {pid}...")
            result = subprocess.run(
                [
                    "criu", "dump",
                    "--tree", str(pid),
                    "--images-dir", str(snapshot_path),
                    "--leave-running",
                    "--shell-job",
                    "--tcp-established",
                    "--file-locks",
                ],
                capture_output=True, text=True, timeout=60,
            )

            if result.returncode != 0:
                print(f"[snapshot] CRIU dump failed: {result.stderr}")
                return None

            # Calculate snapshot size
            size_bytes = sum(f.stat().st_size for f in snapshot_path.rglob("*") if f.is_file())

            # Step 3: Record in DB
            with get_session() as session:
                model = SnapshotModel(
                    snapshot_id=snapshot_id,
                    app_name=app_name,
                    function_name=function_name,
                    class_name=class_name,
                    image_tag=image_tag,
                    snapshot_path=str(snapshot_path),
                    gpu_memory_included=gpu_included,
                    size_bytes=size_bytes,
                )
                session.add(model)
                session.commit()

            print(f"[snapshot] Created {snapshot_id} ({size_bytes / 1024 / 1024:.1f}MB, GPU={gpu_included})")
            return snapshot_id

        except subprocess.TimeoutExpired:
            print("[snapshot] Snapshot creation timed out")
            return None
        except Exception as e:
            print(f"[snapshot] Error: {e}")
            return None

    def restore(self, snapshot_id: str) -> Optional[int]:
        """Restore a process from a CRIU snapshot.

        Returns the PID of the restored process, or None on failure.
        """
        if not self.criu_available:
            return None

        with get_session() as session:
            from sqlmodel import select
            snapshot = session.exec(
                select(SnapshotModel).where(SnapshotModel.snapshot_id == snapshot_id)
            ).first()

        if not snapshot:
            print(f"[snapshot] Snapshot {snapshot_id} not found in DB")
            return None

        snapshot_path = Path(snapshot.snapshot_path)
        if not snapshot_path.exists():
            print(f"[snapshot] Snapshot directory missing: {snapshot_path}")
            return None

        try:
            print(f"[snapshot] Restoring from {snapshot_id}...")
            result = subprocess.run(
                [
                    "criu", "restore",
                    "--images-dir", str(snapshot_path),
                    "--shell-job",
                    "--tcp-established",
                    "--file-locks",
                    "--pidfile", "/tmp/criu-restore.pid",
                ],
                capture_output=True, text=True, timeout=30,
            )

            if result.returncode != 0:
                print(f"[snapshot] CRIU restore failed: {result.stderr}")
                return None

            # Read restored PID
            pid_file = Path("/tmp/criu-restore.pid")
            if pid_file.exists():
                pid = int(pid_file.read_text().strip())
                print(f"[snapshot] Restored PID {pid} from {snapshot_id}")
                return pid

            return None

        except subprocess.TimeoutExpired:
            print("[snapshot] Restore timed out")
            return None

    def delete(self, snapshot_id: str):
        """Delete a snapshot and its files."""
        with get_session() as session:
            from sqlmodel import select
            snapshot = session.exec(
                select(SnapshotModel).where(SnapshotModel.snapshot_id == snapshot_id)
            ).first()
            if snapshot:
                import shutil
                shutil.rmtree(snapshot.snapshot_path, ignore_errors=True)
                session.delete(snapshot)
                session.commit()
