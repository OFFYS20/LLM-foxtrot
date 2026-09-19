"""Typed errors.

One failed operation must never take the application down, so every layer
raises one of these and the UI renders the message instead of a traceback.
"""

from __future__ import annotations


class StudioError(Exception):
    """Base class for every error raised deliberately."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def display(self) -> str:
        return f"{self.message}\n\nHint: {self.hint}" if self.hint else self.message


class ConfigError(StudioError):
    """Invalid configuration."""


class NotFoundError(StudioError):
    """A requested record does not exist."""


class ValidationError(StudioError):
    """User input failed validation."""


class DependencyMissingError(StudioError):
    """An optional dependency is required for this operation.

    Raised instead of silently degrading, so the UI can say exactly what to
    install rather than pretending the feature ran.
    """

    def __init__(self, package: str, feature: str, *, install: str | None = None) -> None:
        super().__init__(
            f"{feature} requires the '{package}' package, which is not installed.",
            hint=f"Install it with: pip install {install or package}",
        )
        self.package = package
        self.feature = feature


class IngestionError(StudioError):
    """A document could not be read or parsed."""


class TokenizerError(StudioError):
    """Tokenizer training or loading failed."""


class TrainingError(StudioError):
    """Training failed."""


class OutOfMemoryError(StudioError):
    """CUDA/host OOM, surfaced with actionable suggestions."""

    def __init__(self, message: str, suggestions: list[str] | None = None) -> None:
        super().__init__(message, hint="; ".join(suggestions or []))
        self.suggestions = suggestions or []


class CheckpointError(StudioError):
    """A checkpoint is missing or corrupt."""


class InferenceError(StudioError):
    """Generation failed."""


class UnsupportedError(StudioError):
    """The requested combination is not supported by this model/architecture."""


class WebError(StudioError):
    """A page or search could not be fetched from the web.

    Separate from IngestionError because the cause is outside this machine —
    no network, a refusing site, a rate limit — and the advice differs.
    """
