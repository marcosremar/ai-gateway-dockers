"""App management endpoints — CRUD + deploy."""

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone

from ..db import get_session, AppModel, AppStatus, ContainerModel
from sqlmodel import select

router = APIRouter(prefix="/v1/apps", tags=["apps"])


class AppCreate(BaseModel):
    name: str
    spec: dict


class AppResponse(BaseModel):
    name: str
    status: str
    functions: list[str]
    classes: list[str]
    created_at: str


class DeployRequest(BaseModel):
    spec: dict
    force_rebuild: bool = False


@router.post("", status_code=201)
async def create_app(body: AppCreate) -> AppResponse:
    """Register a new app."""
    with get_session() as session:
        existing = session.exec(select(AppModel).where(AppModel.name == body.name)).first()
        if existing:
            raise HTTPException(400, f"App '{body.name}' already exists")

        app = AppModel(name=body.name, status=AppStatus.ACTIVE)
        app.spec = body.spec
        session.add(app)
        session.commit()
        session.refresh(app)

        return AppResponse(
            name=app.name,
            status=app.status,
            functions=list(body.spec.get("functions", {}).keys()),
            classes=list(body.spec.get("classes", {}).keys()),
            created_at=app.created_at.isoformat(),
        )


@router.get("")
async def list_apps() -> list[AppResponse]:
    """List all registered apps."""
    with get_session() as session:
        apps = session.exec(select(AppModel)).all()
        return [
            AppResponse(
                name=a.name,
                status=a.status,
                functions=list(a.spec.get("functions", {}).keys()),
                classes=list(a.spec.get("classes", {}).keys()),
                created_at=a.created_at.isoformat(),
            )
            for a in apps
        ]


@router.get("/{app_name}")
async def get_app(app_name: str) -> dict:
    """Get app details including spec."""
    with get_session() as session:
        app = session.exec(select(AppModel).where(AppModel.name == app_name)).first()
        if not app:
            raise HTTPException(404, f"App '{app_name}' not found")

        # Count containers
        containers = session.exec(
            select(ContainerModel).where(ContainerModel.app_name == app_name)
        ).all()

        return {
            "name": app.name,
            "status": app.status,
            "spec": app.spec,
            "containers": {
                "total": len(containers),
                "running": len([c for c in containers if c.status == "running"]),
                "idle": len([c for c in containers if c.status == "idle"]),
            },
            "created_at": app.created_at.isoformat(),
            "updated_at": app.updated_at.isoformat(),
        }


@router.delete("/{app_name}")
async def delete_app(app_name: str):
    """Delete an app and stop all its containers."""
    with get_session() as session:
        app = session.exec(select(AppModel).where(AppModel.name == app_name)).first()
        if not app:
            raise HTTPException(404, f"App '{app_name}' not found")

        # Mark containers for cleanup
        containers = session.exec(
            select(ContainerModel).where(ContainerModel.app_name == app_name)
        ).all()
        for c in containers:
            c.status = "stopping"
            session.add(c)

        session.delete(app)
        session.commit()

    return {"status": "deleted", "app": app_name, "containers_stopped": len(containers)}


@router.post("/{app_name}/deploy")
async def deploy_app(app_name: str, body: DeployRequest) -> dict:
    """Deploy or update an app with new spec."""
    with get_session() as session:
        app = session.exec(select(AppModel).where(AppModel.name == app_name)).first()
        if not app:
            # Auto-create
            app = AppModel(name=app_name, status=AppStatus.DEPLOYING)

        app.spec = body.spec
        app.status = AppStatus.DEPLOYING
        app.updated_at = datetime.now(timezone.utc)
        session.add(app)
        session.commit()

    # TODO: Trigger actual image build + container pool warmup via scheduler
    # For now, mark as active immediately
    with get_session() as session:
        app = session.exec(select(AppModel).where(AppModel.name == app_name)).first()
        if app:
            app.status = AppStatus.ACTIVE
            app.updated_at = datetime.now(timezone.utc)
            session.add(app)
            session.commit()

    return {
        "status": "deployed",
        "app": app_name,
        "functions": list(body.spec.get("functions", {}).keys()),
        "classes": list(body.spec.get("classes", {}).keys()),
    }
