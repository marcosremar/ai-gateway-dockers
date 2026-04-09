"""Warm container pool with scale-to-zero and snapshot-based fast starts."""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
from ..db import get_session, ContainerModel, ContainerStatus, SnapshotModel
from sqlmodel import select


@dataclass
class ContainerInfo:
    """In-memory representation of a running container."""
    container_id: str
    app_name: str
    function_name: Optional[str]
    class_name: Optional[str]
    gpu_device: Optional[str]
    status: str
    last_request_at: float = 0.0
    request_count: int = 0


class ContainerPool:
    """Manages warm containers and handles scale-up/down decisions.

    Lifecycle:
        1. Request arrives → pool.acquire(app, fn)
        2. If warm container available → return immediately
        3. If snapshot exists → CRIU restore (<2s)
        4. Else → cold start (build + start container)
        5. After idle_timeout → snapshot + stop container
    """

    def __init__(
        self,
        idle_timeout: int = 300,
        snapshot_dir: str = "/var/snapgpu/snapshots",
    ):
        self.idle_timeout = idle_timeout
        self.snapshot_dir = snapshot_dir
        self._containers: dict[str, ContainerInfo] = {}  # container_id → info
        self._app_containers: dict[str, list[str]] = {}  # app_name → [container_ids]

    async def acquire(
        self,
        app_name: str,
        function_name: Optional[str] = None,
        class_name: Optional[str] = None,
    ) -> ContainerInfo:
        """Get a ready container for the given function/class.

        Order of preference:
        1. Warm idle container for this function
        2. Restore from snapshot
        3. Cold start new container
        """
        # 1. Check warm pool
        container = self._find_warm(app_name, function_name, class_name)
        if container:
            container.status = "running"
            container.last_request_at = time.time()
            container.request_count += 1
            return container

        # 2. Check for snapshot
        snapshot = await self._find_snapshot(app_name, function_name, class_name)
        if snapshot:
            container = await self._restore_from_snapshot(snapshot)
            if container:
                return container

        # 3. Cold start
        return await self._cold_start(app_name, function_name, class_name)

    def release(self, container_id: str):
        """Mark container as idle after request completes."""
        info = self._containers.get(container_id)
        if info:
            info.status = "idle"
            info.last_request_at = time.time()

    async def cleanup_idle(self):
        """Stop containers that have been idle longer than timeout."""
        now = time.time()
        to_stop = []
        for cid, info in self._containers.items():
            if info.status == "idle" and (now - info.last_request_at) > self.idle_timeout:
                to_stop.append(cid)

        for cid in to_stop:
            await self._stop_container(cid, create_snapshot=True)

    def _find_warm(
        self, app_name: str, function_name: Optional[str], class_name: Optional[str]
    ) -> Optional[ContainerInfo]:
        """Find an idle warm container for this function."""
        candidates = self._app_containers.get(app_name, [])
        for cid in candidates:
            info = self._containers.get(cid)
            if not info or info.status != "idle":
                continue
            if function_name and info.function_name != function_name:
                continue
            if class_name and info.class_name != class_name:
                continue
            return info
        return None

    async def _find_snapshot(
        self, app_name: str, function_name: Optional[str], class_name: Optional[str]
    ) -> Optional[SnapshotModel]:
        """Find the latest snapshot for this function/class."""
        with get_session() as session:
            query = select(SnapshotModel).where(SnapshotModel.app_name == app_name)
            if function_name:
                query = query.where(SnapshotModel.function_name == function_name)
            if class_name:
                query = query.where(SnapshotModel.class_name == class_name)
            query = query.order_by(SnapshotModel.created_at.desc())  # type: ignore
            return session.exec(query).first()

    async def _restore_from_snapshot(self, snapshot: SnapshotModel) -> Optional[ContainerInfo]:
        """Restore a container from a CRIU snapshot."""
        # TODO: Implement actual CRIU restore
        # For now, this is a stub that will be replaced with real implementation
        print(f"[pool] Would restore snapshot {snapshot.snapshot_id} from {snapshot.snapshot_path}")
        return None  # Fall through to cold start

    async def _cold_start(
        self, app_name: str, function_name: Optional[str], class_name: Optional[str]
    ) -> ContainerInfo:
        """Start a new container from scratch (slow path)."""
        # TODO: Use Docker SDK to start container with GPU
        container_id = f"snap-{app_name}-{int(time.time())}"
        info = ContainerInfo(
            container_id=container_id,
            app_name=app_name,
            function_name=function_name,
            class_name=class_name,
            gpu_device=None,
            status="running",
            last_request_at=time.time(),
            request_count=1,
        )
        self._containers[container_id] = info
        if app_name not in self._app_containers:
            self._app_containers[app_name] = []
        self._app_containers[app_name].append(container_id)

        # Persist to DB
        with get_session() as session:
            model = ContainerModel(
                container_id=container_id,
                app_name=app_name,
                function_name=function_name,
                class_name=class_name,
                status=ContainerStatus.RUNNING,
            )
            session.add(model)
            session.commit()

        return info

    async def _stop_container(self, container_id: str, create_snapshot: bool = True):
        """Stop a container, optionally creating a snapshot first."""
        info = self._containers.get(container_id)
        if not info:
            return

        if create_snapshot:
            # TODO: CRIU checkpoint before stopping
            print(f"[pool] Would create snapshot for {container_id} before stopping")

        # TODO: Docker stop
        del self._containers[container_id]
        if info.app_name in self._app_containers:
            self._app_containers[info.app_name] = [
                c for c in self._app_containers[info.app_name] if c != container_id
            ]

        # Update DB
        with get_session() as session:
            model = session.exec(
                select(ContainerModel).where(ContainerModel.container_id == container_id)
            ).first()
            if model:
                model.status = ContainerStatus.STOPPED
                session.add(model)
                session.commit()

    @property
    def stats(self) -> dict:
        """Pool statistics."""
        running = sum(1 for c in self._containers.values() if c.status == "running")
        idle = sum(1 for c in self._containers.values() if c.status == "idle")
        return {
            "total": len(self._containers),
            "running": running,
            "idle": idle,
            "apps": len(self._app_containers),
        }
