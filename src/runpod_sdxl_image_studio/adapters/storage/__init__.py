"""Local persistence adapters."""

from .exceptions import StorageError
from .local_storage import LocalStorageAdapter

__all__ = ["LocalStorageAdapter", "StorageError"]
