"""CRIU-based container snapshots for fast cold starts.

Creates a snapshot after the model is loaded (@enter(snap=True)),
so subsequent starts restore from snapshot instead of re-loading.

Requires CRIU and optionally cuda-checkpoint for GPU memory snapshots.
"""

from __future__ import annotations
import io
import json
import os
import subprocess
import tarfile
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from ..db import get_session, SnapshotModel


# ── S3 config ────────────────────────────────────────────────────────────────

def _s3_client():
    """Return a boto3 S3 client configured from env vars, or None if not set.

    Required env vars:
        SNAPGPU_S3_ENDPOINT    e.g. https://<account>.r2.cloudflarestorage.com
        SNAPGPU_S3_BUCKET      bucket name
        SNAPGPU_S3_ACCESS_KEY  access key ID
        SNAPGPU_S3_SECRET_KEY  secret access key

    Optional:
        SNAPGPU_S3_REGION      default 'auto' (correct for R2; use region for AWS)
        SNAPGPU_S3_KEY_PREFIX  default 'snapgpu/'
    """
    endpoint = os.environ.get("SNAPGPU_S3_ENDPOINT", "")
    bucket = os.environ.get("SNAPGPU_S3_BUCKET", "")
    access_key = os.environ.get("SNAPGPU_S3_ACCESS_KEY", "")
    secret_key = os.environ.get("SNAPGPU_S3_SECRET_KEY", "")
    if not (endpoint and bucket and access_key and secret_key):
        return None, None
    try:
        import boto3
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=os.environ.get("SNAPGPU_S3_REGION", "auto"),
        )
        return client, bucket
    except Exception as e:
        print(f"[snapshot] S3 client init failed: {e}")
        return None, None


def _s3_key_prefix() -> str:
    return os.environ.get("SNAPGPU_S3_KEY_PREFIX", "snapgpu/").rstrip("/") + "/"


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
                print(f"[snapshot] CRIU dump failed (rc={result.returncode}):")
                if result.stdout: print(f"  stdout: {result.stdout.strip()}")
                if result.stderr: print(f"  stderr: {result.stderr.strip()}")
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

            # Auto-upload to S3 if configured (background — don't block inference)
            self._upload_to_s3(snapshot_id, snapshot_path, app_name)

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

    # ── S3 persistence ───────────────────────────────────────────────────────

    def _upload_to_s3(self, snapshot_id: str, snapshot_path: Path, app_name: str) -> bool:
        """Compress snapshot dir and upload to S3 as latest + versioned tar.gz.

        Key schema:
            {prefix}{app_name}/latest.tar.gz        ← always newest
            {prefix}{app_name}/{snapshot_id}.tar.gz ← versioned backup
            {prefix}{app_name}/manifest.json        ← metadata for quick lookup
        """
        s3, bucket = _s3_client()
        if not s3:
            return False

        prefix = _s3_key_prefix()
        try:
            # Compress snapshot dir in-memory
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                tar.add(str(snapshot_path), arcname=snapshot_id)
            compressed = buf.getvalue()
            size_mb = len(compressed) / 1024 / 1024
            print(f"[snapshot] Uploading {snapshot_id} to S3 ({size_mb:.1f}MB)...")

            versioned_key = f"{prefix}{app_name}/{snapshot_id}.tar.gz"
            latest_key = f"{prefix}{app_name}/latest.tar.gz"

            s3.put_object(Bucket=bucket, Key=versioned_key, Body=compressed, ContentType="application/gzip")
            s3.put_object(Bucket=bucket, Key=latest_key, Body=compressed, ContentType="application/gzip")

            # Write manifest
            manifest = {
                "snapshot_id": snapshot_id,
                "app_name": app_name,
                "size_bytes": len(compressed),
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
            s3.put_object(
                Bucket=bucket,
                Key=f"{prefix}{app_name}/manifest.json",
                Body=json.dumps(manifest).encode(),
                ContentType="application/json",
            )
            print(f"[snapshot] S3 upload complete: {latest_key} ({size_mb:.1f}MB)")
            return True
        except Exception as e:
            print(f"[snapshot] S3 upload failed: {e}")
            return False

    def download_from_s3(self, app_name: str) -> Optional[str]:
        """Download latest snapshot from S3, extract locally, register in DB.

        Returns the local snapshot_id on success, None if unavailable.
        Called by start.sh / lifespan before the first restore request.
        """
        s3, bucket = _s3_client()
        if not s3:
            return None

        prefix = _s3_key_prefix()
        manifest_key = f"{prefix}{app_name}/manifest.json"
        latest_key = f"{prefix}{app_name}/latest.tar.gz"

        try:
            # Read manifest to get snapshot_id
            resp = s3.get_object(Bucket=bucket, Key=manifest_key)
            manifest = json.loads(resp["Body"].read())
            remote_snapshot_id = manifest["snapshot_id"]

            # Check if already downloaded locally
            with get_session() as session:
                from sqlmodel import select
                existing = session.exec(
                    select(SnapshotModel).where(SnapshotModel.snapshot_id == remote_snapshot_id)
                ).first()
            if existing and Path(existing.snapshot_path).exists():
                print(f"[snapshot] S3 snapshot {remote_snapshot_id} already local — skipping download")
                return remote_snapshot_id

            print(f"[snapshot] Downloading snapshot {remote_snapshot_id} from S3...")
            resp = s3.get_object(Bucket=bucket, Key=latest_key)
            compressed = resp["Body"].read()
            size_mb = len(compressed) / 1024 / 1024
            print(f"[snapshot] Downloaded {size_mb:.1f}MB — extracting...")

            # Extract to snapshot dir
            buf = io.BytesIO(compressed)
            with tarfile.open(fileobj=buf, mode="r:gz") as tar:
                tar.extractall(str(self.snapshot_dir))

            snapshot_path = self.snapshot_dir / remote_snapshot_id
            if not snapshot_path.exists():
                print(f"[snapshot] Extract failed: {snapshot_path} not found")
                return None

            size_bytes = sum(f.stat().st_size for f in snapshot_path.rglob("*") if f.is_file())

            # Register in local DB
            with get_session() as session:
                model = SnapshotModel(
                    snapshot_id=remote_snapshot_id,
                    app_name=app_name,
                    snapshot_path=str(snapshot_path),
                    size_bytes=size_bytes,
                    gpu_memory_included=True,
                )
                session.add(model)
                session.commit()

            print(f"[snapshot] S3 snapshot {remote_snapshot_id} ready for restore")
            return remote_snapshot_id

        except s3.exceptions.NoSuchKey:
            print(f"[snapshot] No S3 snapshot found for app '{app_name}' — cold start")
            return None
        except Exception as e:
            print(f"[snapshot] S3 download failed: {e}")
            return None

    def s3_manifest(self, app_name: str) -> Optional[dict]:
        """Return the S3 manifest for an app, or None if not found."""
        s3, bucket = _s3_client()
        if not s3:
            return None
        prefix = _s3_key_prefix()
        try:
            resp = s3.get_object(Bucket=bucket, Key=f"{prefix}{app_name}/manifest.json")
            return json.loads(resp["Body"].read())
        except Exception:
            return None
