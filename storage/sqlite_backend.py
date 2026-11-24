"""
SQLite Storage Backend Implementation.
Provides reliable local storage and offline testing capability.
"""
import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

from storage.base import StorageBackend


class SQLiteStorageBackend(StorageBackend):
    def __init__(self, db_path: str = "plagiarism_history.db"):
        self.db_path = db_path
        self.init()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        with self._get_conn() as conn:
            # 1. Scans table (exact match to original app.py schema)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_date TEXT NOT NULL,
                    document_name TEXT,
                    total_sentences INTEGER,
                    exact_matches INTEGER,
                    near_exact_matches INTEGER,
                    semantic_matches INTEGER,
                    verified_sources INTEGER,
                    sources_with_matches INTEGER,
                    overall_similarity REAL,
                    exact_pct REAL,
                    semantic_pct REAL,
                    details TEXT,
                    status TEXT
                )
            """)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(scans)")}
            if "status" not in columns:
                conn.execute("ALTER TABLE scans ADD COLUMN status TEXT")

            # 2. Local Reference Documents
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reference_documents (
                    id TEXT PRIMARY KEY,
                    document_name TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    metadata TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'ready'
                )
            """)

            # 3. Local Reference Chunks
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reference_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    sentence_indices TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB,
                    FOREIGN KEY (document_id) REFERENCES reference_documents(id) ON DELETE CASCADE
                )
            """)

            # 4. AI Detection Scans
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_scans (
                    id TEXT PRIMARY KEY,
                    scanned_at TEXT NOT NULL,
                    document_name TEXT,
                    total_sentences INTEGER,
                    ai_percentage REAL,
                    overall_classification TEXT,
                    details TEXT
                )
            """)

            # 5. Dataset Evaluation Experiments
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dataset_evaluations (
                    id TEXT PRIMARY KEY,
                    evaluated_at TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    dataset_fingerprint TEXT,
                    sample_count INTEGER,
                    models_evaluated TEXT,
                    best_model TEXT,
                    best_f1 REAL,
                    best_accuracy REAL,
                    best_roc_auc REAL,
                    execution_time_seconds REAL,
                    details TEXT
                )
            """)

    def save_scan(self, result: Dict[str, Any]) -> int:
        s = result["stats"]
        incomplete = result.get("status") == "incomplete"
        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO scans (scan_date, document_name, total_sentences, exact_matches,
                   near_exact_matches, semantic_matches, verified_sources, sources_with_matches,
                   overall_similarity, exact_pct, semantic_pct, details, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (result.get("scanned_at", time.strftime("%Y-%m-%d %H:%M:%S")),
                 result.get("document_name", "Untitled"),
                 s.get("total_sentences", 0),
                 s.get("exact", 0),
                 s.get("near_exact", 0),
                 s.get("semantic", 0),
                 s.get("verified_sources", 0),
                 s.get("sources_with_matches", 0),
                 None if incomplete else s.get("overall_pct", 0.0),
                 None if incomplete else s.get("exact_pct", 0.0),
                 None if incomplete else s.get("semantic_pct", 0.0),
                 json.dumps(result),
                 result.get("status"))
            )
            return int(cursor.lastrowid)

    def get_scan_history(self, limit: int = 200) -> List[Dict[str, Any]]:
        legacy_incomplete = "(status IS NULL AND verified_sources = 0)"
        hide = f"(status = 'incomplete' OR {legacy_incomplete})"
        with self._get_conn() as conn:
            rows = conn.execute(
                f"""SELECT id AS "Scan ID", scan_date AS "Date/Time", document_name AS "Document",
                   CASE WHEN status IS NOT NULL THEN status
                        WHEN verified_sources = 0 THEN 'incomplete (older version)'
                        ELSE 'completed (older version)' END AS "Status",
                   total_sentences AS "Sentences", exact_matches AS "Exact",
                   near_exact_matches AS "Near-exact", semantic_matches AS "Semantic",
                   verified_sources AS "Verified sources", sources_with_matches AS "Sources with matches",
                   CASE WHEN {hide} THEN NULL ELSE overall_similarity END AS "Overall %",
                   CASE WHEN {hide} THEN NULL ELSE exact_pct END AS "Exact/near %",
                   CASE WHEN {hide} THEN NULL ELSE semantic_pct END AS "Semantic %"
                   FROM scans ORDER BY id DESC LIMIT ?""", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_scan_details(self, scan_id: int) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT details FROM scans WHERE id = ?", (scan_id,)).fetchone()
        return json.loads(row["details"]) if row and row["details"] else None

    def save_reference_document(
        self,
        document_name: str,
        metadata: Dict[str, Any],
        chunks: List[Dict[str, Any]]
    ) -> str:
        import uuid
        doc_id = str(uuid.uuid4())
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO reference_documents (id, document_name, uploaded_at, metadata, chunk_count, status)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (doc_id, document_name, now, json.dumps(metadata), len(chunks), "ready")
            )
            for idx, ch in enumerate(chunks):
                ch_id = f"{doc_id}_{idx}"
                # Store vector embedding as JSON or float bytes
                emb = ch.get("embedding")
                emb_blob = json.dumps(emb if isinstance(emb, list) else emb.tolist()).encode("utf-8") if emb is not None else None
                sent_indices = json.dumps(ch.get("sentence_indices", [idx]))
                conn.execute(
                    """INSERT INTO reference_chunks (id, document_id, chunk_index, sentence_indices, text, embedding)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (ch_id, doc_id, idx, sent_indices, ch.get("text", ""), emb_blob)
                )
        return doc_id

    def get_reference_documents(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM reference_documents ORDER BY uploaded_at DESC").fetchall()
            docs = []
            for r in rows:
                d = dict(r)
                if d.get("metadata"):
                    try:
                        d["metadata"] = json.loads(d["metadata"])
                    except Exception:
                        pass
                docs.append(d)
            return docs

    def delete_reference_document(self, doc_id: str) -> bool:
        with self._get_conn() as conn:
            conn.execute("DELETE FROM reference_chunks WHERE document_id = ?", (doc_id,))
            cursor = conn.execute("DELETE FROM reference_documents WHERE id = ?", (doc_id,))
            return cursor.rowcount > 0

    def get_reference_chunks(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            if doc_id:
                query = """SELECT rc.*, rd.document_name
                           FROM reference_chunks rc
                           JOIN reference_documents rd ON rc.document_id = rd.id
                           WHERE rc.document_id = ?
                           ORDER BY rc.chunk_index ASC"""
                rows = conn.execute(query, (doc_id,)).fetchall()
            else:
                query = """SELECT rc.*, rd.document_name
                           FROM reference_chunks rc
                           JOIN reference_documents rd ON rc.document_id = rd.id
                           ORDER BY rc.document_id, rc.chunk_index ASC"""
                rows = conn.execute(query).fetchall()

            chunks = []
            for r in rows:
                ch = dict(r)
                ch["sentence_indices"] = json.loads(ch["sentence_indices"])
                if ch.get("embedding"):
                    ch["embedding"] = json.loads(ch["embedding"].decode("utf-8"))
                chunks.append(ch)
            return chunks

    def save_ai_scan(
        self,
        document_name: str,
        sentence_results: List[Dict[str, Any]],
        summary: Dict[str, Any]
    ) -> str:
        import uuid
        scan_id = str(uuid.uuid4())
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        details = {
            "scan_id": scan_id,
            "document_name": document_name,
            "scanned_at": now,
            "sentences": sentence_results,
            "summary": summary
        }
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO ai_scans (id, scanned_at, document_name, total_sentences,
                   ai_percentage, overall_classification, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (scan_id, now, document_name, len(sentence_results),
                 summary.get("ai_percentage", 0.0), summary.get("overall_classification", ""),
                 json.dumps(details))
            )
        return scan_id

    def get_ai_scan_details(self, scan_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT details FROM ai_scans WHERE id = ?", (scan_id,)).fetchone()
        return json.loads(row["details"]) if row and row["details"] else None

    def save_evaluation(self, experiment_data: Dict[str, Any]) -> str:
        import uuid
        exp_id = experiment_data.get("experiment_id") or f"exp_{int(time.time())}_{str(uuid.uuid4())[:6]}"
        now = experiment_data.get("evaluated_at") or time.strftime("%Y-%m-%d %H:%M:%S")
        dataset_name = experiment_data.get("dataset", "Webis-CPC-11")
        fingerprint = experiment_data.get("dataset_fingerprint", "")
        sample_count = experiment_data.get("total_records", 0)

        # Determine best model based on F1
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

        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO dataset_evaluations
                   (id, evaluated_at, dataset_name, dataset_fingerprint, sample_count,
                    models_evaluated, best_model, best_f1, best_accuracy, best_roc_auc,
                    execution_time_seconds, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (exp_id, now, dataset_name, fingerprint, sample_count,
                 ", ".join(models), best_name, best_f1, best_acc, best_auc,
                 experiment_data.get("total_time_seconds", 0.0),
                 json.dumps(experiment_data))
            )
        return exp_id

    def get_evaluations(self, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            if dataset_name:
                rows = conn.execute(
                    "SELECT * FROM dataset_evaluations WHERE dataset_name = ? ORDER BY evaluated_at DESC",
                    (dataset_name,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM dataset_evaluations ORDER BY evaluated_at DESC"
                ).fetchall()

        results = []
        for r in rows:
            results.append({
                "experiment_id": r["id"],
                "evaluated_at": r["evaluated_at"],
                "dataset_name": r["dataset_name"],
                "dataset_fingerprint": r["dataset_fingerprint"],
                "sample_count": r["sample_count"],
                "models_evaluated": r["models_evaluated"],
                "best_model": r["best_model"],
                "best_f1": r["best_f1"],
                "best_accuracy": r["best_accuracy"],
                "best_roc_auc": r["best_roc_auc"],
                "execution_time_seconds": r["execution_time_seconds"],
            })
        return results

    def get_evaluation_details(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT details FROM dataset_evaluations WHERE id = ?",
                (experiment_id,)
            ).fetchone()
        return json.loads(row["details"]) if row and row["details"] else None
