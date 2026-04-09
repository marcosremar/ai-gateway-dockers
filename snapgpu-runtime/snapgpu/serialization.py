"""Function serialization for remote execution."""

from __future__ import annotations
import base64
import hashlib
import pickle
from typing import Any, Callable

try:
    import cloudpickle
    _PICKLE = cloudpickle
except ImportError:
    _PICKLE = pickle  # type: ignore


def serialize_function(fn: Callable) -> bytes:
    """Serialize a function and its closure for remote execution."""
    return _PICKLE.dumps(fn)


def deserialize_function(data: bytes) -> Callable:
    """Deserialize a function from bytes."""
    return _PICKLE.loads(data)


def serialize_args(args: tuple, kwargs: dict) -> bytes:
    """Serialize function arguments."""
    return _PICKLE.dumps((args, kwargs))


def deserialize_args(data: bytes) -> tuple[tuple, dict]:
    """Deserialize function arguments."""
    return _PICKLE.loads(data)


def serialize_result(result: Any) -> bytes:
    """Serialize a function result."""
    return _PICKLE.dumps(result)


def deserialize_result(data: bytes) -> Any:
    """Deserialize a function result."""
    return _PICKLE.loads(data)


def function_hash(fn: Callable) -> str:
    """Compute a content-addressable hash for a function."""
    data = serialize_function(fn)
    return hashlib.sha256(data).hexdigest()[:16]


def encode_b64(data: bytes) -> str:
    """Encode bytes as base64 string (for JSON transport)."""
    return base64.b64encode(data).decode("ascii")


def decode_b64(s: str) -> bytes:
    """Decode base64 string to bytes."""
    return base64.b64decode(s)
