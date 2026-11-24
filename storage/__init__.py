"""Storage backend package for PlagiarismAI.

Provides an abstract StorageBackend interface with two concrete implementations:
- SQLiteStorageBackend  – default, zero-config, file-based
- MongoStorageBackend   – optional, set MONGODB_URI env var to enable

Use `storage.repository.get_storage()` to obtain the active backend.
"""

from storage.base import StorageBackend
from storage.repository import get_storage

__all__ = ["StorageBackend", "get_storage"]
