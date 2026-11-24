"""
Storage Backend Abstraction Layer.
Defines abstract interface for scan history, local reference database, and AI results.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class StorageBackend(ABC):
    @abstractmethod
    def init(self) -> None:
        """Initialize database schema/collections and indexes."""
        pass

    @abstractmethod
    def save_scan(self, result: Dict[str, Any]) -> int:
        """Save a plagiarism scan and return the new scan ID."""
        pass

    @abstractmethod
    def get_scan_history(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Return previous scans newest first."""
        pass

    @abstractmethod
    def get_scan_details(self, scan_id: int) -> Optional[Dict[str, Any]]:
        """Return full details JSON for a scan ID."""
        pass

    @abstractmethod
    def save_reference_document(
        self,
        document_name: str,
        metadata: Dict[str, Any],
        chunks: List[Dict[str, Any]]
    ) -> str:
        """Save a reference document and its indexed chunks with embeddings."""
        pass

    @abstractmethod
    def get_reference_documents(self) -> List[Dict[str, Any]]:
        """List all stored reference documents."""
        pass

    @abstractmethod
    def delete_reference_document(self, doc_id: str) -> bool:
        """Safely delete a reference document and its chunks."""
        pass

    @abstractmethod
    def get_reference_chunks(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve reference chunks (including embeddings) for similarity comparison."""
        pass

    @abstractmethod
    def save_ai_scan(
        self,
        document_name: str,
        sentence_results: List[Dict[str, Any]],
        summary: Dict[str, Any]
    ) -> str:
        """Save sentence-level AI detection results."""
        pass

    @abstractmethod
    def get_ai_scan_details(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve AI scan details."""
        pass

    @abstractmethod
    def save_evaluation(self, experiment_data: Dict[str, Any]) -> str:
        """Save benchmark/evaluation experiment results and return experiment ID."""
        pass

    @abstractmethod
    def get_evaluations(self, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List previous evaluation experiment summaries."""
        pass

    @abstractmethod
    def get_evaluation_details(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full details of an evaluation experiment."""
        pass
