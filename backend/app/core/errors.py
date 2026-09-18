"""Typed application errors mapped to HTTP responses by ``app.main``."""

from __future__ import annotations

from typing import Any


class FoxtrotError(Exception):
    """Base class for every error the platform raises deliberately."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_payload(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class NotFoundError(FoxtrotError):
    status_code = 404
    code = "not_found"


class ValidationError(FoxtrotError):
    status_code = 422
    code = "validation_error"


class ConflictError(FoxtrotError):
    status_code = 409
    code = "conflict"


class UnsupportedError(FoxtrotError):
    """A capability exists in the interface but not in this deployment."""

    status_code = 501
    code = "unsupported"


class EngineUnavailableError(FoxtrotError):
    """A backend engine (torch, nvml, llama.cpp …) is not installed/usable."""

    status_code = 503
    code = "engine_unavailable"


class OutOfMemoryError(FoxtrotError):
    """CUDA / host OOM surfaced cleanly instead of crashing the worker."""

    status_code = 507
    code = "out_of_memory"


class PathTraversalError(FoxtrotError):
    status_code = 400
    code = "path_traversal"
