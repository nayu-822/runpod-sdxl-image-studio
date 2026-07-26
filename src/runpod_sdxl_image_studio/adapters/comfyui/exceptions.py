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


class ComfyUIParseError(ComfyUIError):
    """A valid response did not contain an interpretable structure."""
