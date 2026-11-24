"""
Storage Repository Factory.

Provides a unified interface that defaults to MongoDB when configured,
with automatic graceful fallback to SQLite for offline development and testing.
"""
import os
from typing import Optional

from storage.base import StorageBackend
from storage.sqlite_backend import SQLiteStorageBackend


_GLOBAL_STORAGE: Optional[StorageBackend] = None


def get_storage(
    db_path: str = "plagiarism_history.db",
    prefer_mongo: bool = True,
    force_refresh: bool = False
) -> StorageBackend:
    """
    Get the active storage backend.
    
    1. Tries MongoDB if prefer_mongo is True and MONGODB_URI is provided.
    2. Gracefully falls back to SQLite if MongoDB is unavailable or fails to connect.
    """
    global _GLOBAL_STORAGE
    if _GLOBAL_STORAGE is not None and not force_refresh:
        return _GLOBAL_STORAGE

    if prefer_mongo and os.getenv("MONGODB_URI"):
        try:
            from storage.mongo_backend import MongoStorageBackend
            backend = MongoStorageBackend()
            _GLOBAL_STORAGE = backend
            return backend
        except Exception as e:
            print(f"[Storage] MongoDB connection unavailable ({e}); falling back to SQLite ({db_path}).")

    backend = SQLiteStorageBackend(db_path=db_path)
    _GLOBAL_STORAGE = backend
    return backend
