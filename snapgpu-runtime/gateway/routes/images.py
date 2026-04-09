"""Image management endpoints."""

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..db import get_session, ImageModel
from sqlmodel import select

router = APIRouter(prefix="/v1/images", tags=["images"])


@router.get("")
async def list_images() -> list[dict]:
    """List all built images."""
    with get_session() as session:
        images = session.exec(select(ImageModel)).all()
        return [
            {
                "tag": img.tag,
                "app_name": img.app_name,
                "size_mb": round(img.size_bytes / 1024 / 1024, 1),
                "built_at": img.built_at.isoformat() if img.built_at else None,
            }
            for img in images
        ]


@router.delete("/{tag}")
async def delete_image(tag: str):
    """Delete a built image."""
    with get_session() as session:
        image = session.exec(select(ImageModel).where(ImageModel.tag == tag)).first()
        if not image:
            raise HTTPException(404, f"Image '{tag}' not found")
        session.delete(image)
        session.commit()
    return {"status": "deleted", "tag": tag}
