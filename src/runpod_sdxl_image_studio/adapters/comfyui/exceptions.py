"""Exceptions raised while communicating with or parsing ComfyUI."""

from __future__ import annotations


class ComfyUIError(Exception):
    """Base exception for ComfyUI adapter failures."""


class ComfyUIConnectionError(ComfyUIError):
    """The ComfyUI endpoint could not be reached."""


class ComfyUITimeoutError(ComfyUIError):
    """The ComfyUI endpoint did not respond before the configured timeout."""


class ComfyUIResponseError(ComfyUIError):
    """ComfyUI returned an HTTP or JSON response that cannot be used."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ComfyUIParseError(ComfyUIError):
    """A valid response did not contain an interpretable structure."""


class ComfyUIPromptError(ComfyUIError):
    """ComfyUI rejected a queued prompt or reported node errors."""


class ComfyUIWebSocketError(ComfyUIError):
    """The ComfyUI WebSocket could not be monitored safely."""


class ComfyUIWebSocketTimeoutError(ComfyUIWebSocketError):
    """The WebSocket did not produce a message before its timeout."""


class ComfyUIWebSocketDisconnectedError(ComfyUIWebSocketError):
    """The WebSocket disconnected before the prompt was confirmed complete."""


class WorkflowError(Exception):
    """Base exception for fixed workflow validation failures."""


class WorkflowTemplateError(WorkflowError):
    """The repository-controlled workflow template is invalid."""


class WorkflowBindingError(WorkflowError):
    """A fixed binding path does not exist in the workflow."""


class WorkflowValidationError(WorkflowError):
    """Workflow values cannot be executed safely."""
