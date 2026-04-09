"""Docker image builder with layer caching."""

from __future__ import annotations
import hashlib
import tempfile
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from ..db import get_session, ImageModel


class ImageBuilder:
    """Builds Docker images from SnapGPU Image specs.

    Features:
    - Layer-level caching (same pip_install → same layer hash)
    - Deduplication: same Dockerfile content → skip build
    - Tracks built images in DB for cleanup
    """

    def __init__(self, registry_prefix: str = "snapgpu"):
        self.registry_prefix = registry_prefix
        self._docker_client = None

    @property
    def docker(self):
        if self._docker_client is None:
            try:
                import docker
                self._docker_client = docker.from_env()
            except ImportError:
                raise ImportError("docker SDK is required: pip install docker")
        return self._docker_client

    def build(self, app_name: str, dockerfile_content: str,
              function_name: Optional[str] = None, force: bool = False) -> str:
        """Build a Docker image from Dockerfile content.

        Returns the image tag.
        """
        # Content-addressable tag
        content_hash = hashlib.sha256(dockerfile_content.encode()).hexdigest()[:12]
        tag = f"{self.registry_prefix}/{app_name}:{content_hash}"

        # Check if already built (dedup)
        if not force:
            with get_session() as session:
                from sqlmodel import select
                existing = session.exec(
                    select(ImageModel).where(ImageModel.dockerfile_hash == content_hash)
                ).first()
                if existing:
                    print(f"[builder] Image already exists: {tag}")
                    return existing.tag

        # Write Dockerfile to temp dir and build
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile_path = Path(tmpdir) / "Dockerfile"
            dockerfile_path.write_text(dockerfile_content)

            print(f"[builder] Building image {tag}...")
            image, logs = self.docker.images.build(
                path=tmpdir,
                tag=tag,
                rm=True,
                forcerm=True,
            )

            # Get image size
            image.reload()
            size_bytes = image.attrs.get("Size", 0)

        # Record in DB
        with get_session() as session:
            model = ImageModel(
                tag=tag,
                app_name=app_name,
                dockerfile_hash=content_hash,
                size_bytes=size_bytes,
                built_at=datetime.now(timezone.utc),
            )
            session.add(model)
            session.commit()

        print(f"[builder] Built {tag} ({size_bytes / 1024 / 1024:.1f}MB)")
        return tag

    def image_exists(self, tag: str) -> bool:
        """Check if an image exists locally."""
        try:
            self.docker.images.get(tag)
            return True
        except Exception:
            return False

    def cleanup(self, app_name: str):
        """Remove all images for an app."""
        with get_session() as session:
            from sqlmodel import select
            images = session.exec(
                select(ImageModel).where(ImageModel.app_name == app_name)
            ).all()
            for img in images:
                try:
                    self.docker.images.remove(img.tag, force=True)
                except Exception:
                    pass
                session.delete(img)
            session.commit()
