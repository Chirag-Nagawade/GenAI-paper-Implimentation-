"""
MongoDB Storage Backend Implementation.

Implements collections:
- scan_history
- reference_documents
- reference_chunks (one document per chunk with embedding vector)
- plagiarism_results
- ai_detection_results
- dataset_evaluations
"""
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from storage.base import StorageBackend

try:
    from pymongo import ASCENDING, DESCENDING, MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False


class MongoStorageBackend(StorageBackend):
    def __init__(
        self,
        uri: Optional[str] = None,
        db_name: Optional[str] = None,
        timeout_ms: int = 3000
    ):
        if not PYMONGO_AVAILABLE:
            raise ImportError("pymongo is not installed.")

        self.uri = uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.db_name = db_name or os.getenv("MONGODB_DB_NAME", "plagiarism_ai")
        self.timeout_ms = timeout_ms

        self.client = MongoClient(self.uri, serverSelectionTimeoutMS=self.timeout_ms)
        # Verify connection
        self.client.admin.command("ping")
        self.db = self.client[self.db_name]

        self.scan_history = self.db["scan_history"]
        self.reference_documents = self.db["reference_documents"]
        self.reference_chunks = self.db["reference_chunks"]
        self.plagiarism_results = self.db["plagiarism_results"]
        self.ai_detection_results = self.db["ai_detection_results"]
        self.dataset_evaluations = self.db["dataset_evaluations"]

        self.init()

    def init(self) -> None:
        """Create indexes on all primary collections."""
        try:
            self.scan_history.create_index([("scan_id", DESCENDING)], unique=True)
            self.scan_history.create_index([("scanned_at", DESCENDING)])
            self.reference_documents.create_index([("id", ASCENDING)], unique=True)
            self.reference_chunks.create_index([("id", ASCENDING)], unique=True)
            self.reference_chunks.create_index([("document_id", ASCENDING)])
            self.plagiarism_results.create_index([("scan_id", ASCENDING)])
            self.ai_detection_results.create_index([("scan_id", ASCENDING)])
            self.dataset_evaluations.create_index([("dataset_name", ASCENDING)])
        except Exception:
            pass

    def _get_next_scan_id(self) -> int:
        counter_col = self.db["counters"]
        ret = counter_col.find_one_and_update(
            {"_id": "scan_id"},
            {"$inc": {"seq": 1}},
            upsert=True,
            return_document=True
        )
        return int(ret["seq"])

    def save_scan(self, result: Dict[str, Any]) -> int:
        scan_id = result.get("id") or self._get_next_scan_id()
        s = result["stats"]
        incomplete = result.get("status") == "incomplete"

        doc = {
            "scan_id": scan_id,
            "document_name": result.get("document_name", "Untitled"),
            "scan_date": result.get("scanned_at", time.strftime("%Y-%m-%d %H:%M:%S")),
            "status": result.get("status"),
            "total_sentences": s.get("total_sentences", 0),
            "exact_matches": s.get("exact", 0),
            "near_exact_matches": s.get("near_exact", 0),
            "semantic_matches": s.get("semantic", 0),
            "verified_sources": s.get("verified_sources", 0),
            "sources_with_matches": s.get("sources_with_matches", 0),
            "overall_similarity": None if incomplete else s.get("overall_pct", 0.0),
            "exact_pct": None if incomplete else s.get("exact_pct", 0.0),
            "semantic_pct": None if incomplete else s.get("semantic_pct", 0.0),
            "details": result
        }

        self.scan_history.update_one({"scan_id": scan_id}, {"$set": doc}, upsert=True)

        # Store detailed matches in plagiarism_results
        self.plagiarism_results.delete_many({"scan_id": scan_id})
        matches = result.get("matches", [])
        if matches:
            match_docs = []
            for m in matches:
                md = dict(m)
                md["scan_id"] = scan_id
                match_docs.append(md)
            self.plagiarism_results.insert_many(match_docs)

        return scan_id

    def get_scan_history(self, limit: int = 200) -> List[Dict[str, Any]]:
        cursor = self.scan_history.find(
            {},
            {"_id": 0, "details": 0}
        ).sort("scan_id", DESCENDING).limit(limit)

        records = []
        for doc in cursor:
            status = doc.get("status")
            if not status:
                status = "incomplete (older version)" if doc.get("verified_sources", 0) == 0 else "completed (older version)"
            hide = (status == "incomplete" or status == "incomplete (older version)")

            records.append({
                "Scan ID": doc.get("scan_id"),
                "Date/Time": doc.get("scan_date"),
                "Document": doc.get("document_name"),
                "Status": status,
                "Sentences": doc.get("total_sentences", 0),
                "Exact": doc.get("exact_matches", 0),
                "Near-exact": doc.get("near_exact_matches", 0),
                "Semantic": doc.get("semantic_matches", 0),
                "Verified sources": doc.get("verified_sources", 0),
                "Sources with matches": doc.get("sources_with_matches", 0),
                "Overall %": None if hide else doc.get("overall_similarity"),
                "Exact/near %": None if hide else doc.get("exact_pct"),
                "Semantic %": None if hide else doc.get("semantic_pct"),
            })
        return records

    def get_scan_details(self, scan_id: int) -> Optional[Dict[str, Any]]:
        doc = self.scan_history.find_one({"scan_id": int(scan_id)})
        return doc.get("details") if doc else None

    def save_reference_document(
        self,
        document_name: str,
        metadata: Dict[str, Any],
        chunks: List[Dict[str, Any]]
    ) -> str:
        import uuid
        doc_id = str(uuid.uuid4())
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        doc_record = {
            "id": doc_id,
            "document_name": document_name,
            "uploaded_at": now,
            "metadata": metadata,
            "chunk_count": len(chunks),
            "status": "ready"
        }
        self.reference_documents.insert_one(doc_record)

        # Store chunks as individual records
        chunk_docs = []
        for idx, ch in enumerate(chunks):
            emb = ch.get("embedding")
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()
            chunk_docs.append({
                "id": f"{doc_id}_{idx}",
                "document_id": doc_id,
                "document_name": document_name,
                "chunk_index": idx,
                "sentence_indices": ch.get("sentence_indices", [idx]),
                "text": ch.get("text", ""),
                "embedding": emb
            })
        if chunk_docs:
            self.reference_chunks.insert_many(chunk_docs)
        return doc_id

    def get_reference_documents(self) -> List[Dict[str, Any]]:
        return list(self.reference_documents.find({}, {"_id": 0}).sort("uploaded_at", DESCENDING))

    def delete_reference_document(self, doc_id: str) -> bool:
        self.reference_chunks.delete_many({"document_id": doc_id})
        res = self.reference_documents.delete_one({"id": doc_id})
        return res.deleted_count > 0

    def get_reference_chunks(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query = {"document_id": doc_id} if doc_id else {}
        return list(self.reference_chunks.find(query, {"_id": 0}))

    def save_ai_scan(
        self,
        document_name: str,
        sentence_results: List[Dict[str, Any]],
        summary: Dict[str, Any]
    ) -> str:
        import uuid
        scan_id = str(uuid.uuid4())
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        doc = {
            "scan_id": scan_id,
            "document_name": document_name,
            "scanned_at": now,
            "total_sentences": len(sentence_results),
            "ai_percentage": summary.get("ai_percentage", 0.0),
            "overall_classification": summary.get("overall_classification", ""),
            "summary": summary,
            "sentences": sentence_results
        }
        self.ai_detection_results.insert_one(doc)
        return scan_id

    def get_ai_scan_details(self, scan_id: str) -> Optional[Dict[str, Any]]:
        return self.ai_detection_results.find_one({"scan_id": scan_id}, {"_id": 0})

    def save_evaluation(self, experiment_data: Dict[str, Any]) -> str:
        import uuid
        exp_id = experiment_data.get("experiment_id") or f"exp_{int(time.time())}_{str(uuid.uuid4())[:6]}"
        now = experiment_data.get("evaluated_at") or time.strftime("%Y-%m-%d %H:%M:%S")
        dataset_name = experiment_data.get("dataset", "Webis-CPC-11")
        fingerprint = experiment_data.get("dataset_fingerprint", "")
        sample_count = experiment_data.get("total_records", 0)

        models = []
        best_name, best_f1, best_acc, best_auc = "N/A", 0.0, 0.0, 0.0
        for k, v in experiment_data.items():
            if isinstance(v, dict) and "f1_score" in v:
                models.append(k)
                if v.get("f1_score", 0.0) > best_f1:
                    best_f1 = v["f1_score"]
                    best_name = k
                    best_acc = v.get("accuracy", 0.0)
                    best_auc = v.get("roc_auc", 0.0)

        doc = {
            "experiment_id": exp_id,
            "evaluated_at": now,
            "dataset_name": dataset_name,
            "dataset_fingerprint": fingerprint,
            "sample_count": sample_count,
            "models_evaluated": ", ".join(models),
            "best_model": best_name,
            "best_f1": best_f1,
            "best_accuracy": best_acc,
            "best_roc_auc": best_auc,
            "execution_time_seconds": experiment_data.get("total_time_seconds", 0.0),
            "details": experiment_data
        }
        self.dataset_evaluations.update_one({"experiment_id": exp_id}, {"$set": doc}, upsert=True)
        return exp_id

    def get_evaluations(self, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        query = {"dataset_name": dataset_name} if dataset_name else {}
        cursor = self.dataset_evaluations.find(query, {"_id": 0, "details": 0}).sort("evaluated_at", DESCENDING)
        return list(cursor)

    def get_evaluation_details(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        doc = self.dataset_evaluations.find_one({"experiment_id": experiment_id}, {"_id": 0})
        return doc.get("details") if doc else None
