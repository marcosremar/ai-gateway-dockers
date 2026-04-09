"""Function invocation endpoint — the core execution path."""

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..db import get_session, AppModel, TaskModel, TaskStatus
from sqlmodel import select

router = APIRouter(prefix="/v1/invoke", tags=["invoke"])


class InvokeRequest(BaseModel):
    fn_data: Optional[str] = None   # Base64 cloudpickle'd function
    args_data: Optional[str] = None  # Base64 cloudpickle'd (args, kwargs)
    args: Optional[dict] = None      # JSON args (alternative to pickle)
    async_mode: bool = False         # If true, return task_id immediately

    class Config:
        # Allow 'async' as field name in JSON
        populate_by_name = True


class InvokeResponse(BaseModel):
    status: str
    result: Optional[str] = None  # Base64 pickled result
    task_id: Optional[str] = None
    container_id: Optional[str] = None
    latency_ms: Optional[float] = None


@router.post("/{app_name}/{fn_name}")
async def invoke_function(app_name: str, fn_name: str, body: InvokeRequest) -> InvokeResponse:
    """Invoke a registered function on a GPU container."""
    # Validate app exists
    with get_session() as session:
        app = session.exec(select(AppModel).where(AppModel.name == app_name)).first()
        if not app:
            raise HTTPException(404, f"App '{app_name}' not found")

        spec = app.spec
        fn_spec = spec.get("functions", {}).get(fn_name)
        cls_spec = None
        if not fn_spec:
            # Check if it's a class method
            for cls_name, cls_data in spec.get("classes", {}).items():
                if fn_name in cls_data.get("methods", []):
                    cls_spec = cls_data
                    break
            if not cls_spec:
                raise HTTPException(404, f"Function '{fn_name}' not found in app '{app_name}'")

    task_id = str(uuid.uuid4())

    # Async mode: queue and return immediately
    if body.async_mode:
        with get_session() as session:
            task = TaskModel(
                task_id=task_id,
                app_name=app_name,
                function_name=fn_name,
                status=TaskStatus.PENDING,
            )
            session.add(task)
            session.commit()

        # TODO: Submit to task queue (Redis) for async execution
        return InvokeResponse(status="queued", task_id=task_id)

    # Sync mode: execute now
    start = datetime.now(timezone.utc)

    try:
        # TODO: Route to container pool → find/create container → execute
        # For now, execute locally if fn_data is provided
        if body.fn_data:
            from snapgpu.serialization import decode_b64, deserialize_function, deserialize_args, serialize_result, encode_b64
            fn = deserialize_function(decode_b64(body.fn_data))
            args, kwargs = deserialize_args(decode_b64(body.args_data)) if body.args_data else ((), {})
            result = fn(*args, **kwargs)
            result_data = encode_b64(serialize_result(result))
        elif body.args:
            # JSON args mode — function must be pre-loaded in container
            result_data = None  # TODO: Route to container
            raise HTTPException(501, "JSON args invocation requires container routing (not yet implemented)")
        else:
            raise HTTPException(400, "Either fn_data or args must be provided")

        elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        return InvokeResponse(
            status="completed",
            result=result_data,
            latency_ms=elapsed,
        )

    except HTTPException:
        raise
    except Exception as e:
        elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        return InvokeResponse(
            status="error",
            result=None,
            latency_ms=elapsed,
        )


@router.get("/{app_name}/{fn_name}/status/{task_id}")
async def get_task_status(app_name: str, fn_name: str, task_id: str) -> dict:
    """Check the status of an async task."""
    with get_session() as session:
        task = session.exec(select(TaskModel).where(TaskModel.task_id == task_id)).first()
        if not task:
            raise HTTPException(404, f"Task '{task_id}' not found")

        return {
            "task_id": task.task_id,
            "status": task.status,
            "result": task.result_data if task.status == "completed" else None,
            "error": task.error if task.status == "failed" else None,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        }
