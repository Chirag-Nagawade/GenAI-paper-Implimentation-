"""
Webis-CPC-11 Dataset Loader

Loads and validates the Webis-CPC-11 paraphrase corpus.
Dataset consists of original passages, candidate paraphrase passages,
and metadata specifying whether the candidate is a true paraphrase (Yes/No).
"""
import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import joblib


@dataclass
class WebisRecord:
    record_id: str
    original_text: str
    paraphrase_text: str
    label: int  # 1 for Paraphrase: Yes, 0 for Paraphrase: No
    metadata: Dict[str, str] = field(default_factory=dict)


def compute_dataset_fingerprint(records: List[WebisRecord]) -> str:
    """Compute a SHA-256 fingerprint of the dataset record IDs and labels."""
    h = hashlib.sha256()
    for r in records:
        h.update(f"{r.record_id}:{r.label};".encode("utf-8"))
    return h.hexdigest()[:16]


def parse_metadata_file(path: str) -> Dict[str, str]:
    """Parse key-value pairs from a Webis metadata file."""
    meta = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    except OSError:
        pass
    return meta


def _sample_records(
    records: List[WebisRecord],
    max_records: int,
    stratified: bool = True,
    random_seed: int = 42
) -> List[WebisRecord]:
    """Sample records with optional reproducible class stratification."""
    if max_records >= len(records):
        return records
    if not stratified:
        return records[:max_records]

    pos_recs = [r for r in records if r.label == 1]
    neg_recs = [r for r in records if r.label == 0]
    n_pos = int(round(max_records * (len(pos_recs) / len(records))))
    n_neg = max_records - n_pos

    import numpy as np
    rng = np.random.RandomState(random_seed)
    pos_chosen = rng.choice(len(pos_recs), size=n_pos, replace=False)
    neg_chosen = rng.choice(len(neg_recs), size=n_neg, replace=False)

    sampled = [pos_recs[i] for i in sorted(pos_chosen)] + [neg_recs[j] for j in sorted(neg_chosen)]
    rng.shuffle(sampled)
    return sampled


def load_webis_dataset(
    dataset_dir: str = "data/Webis-CPC-11",
    max_records: Optional[int] = None,
    verbose: bool = True,
    use_cache: bool = True,
    stratified: bool = True,
    random_seed: int = 42
) -> Tuple[List[WebisRecord], Dict[str, any]]:
    """
    Load Webis-CPC-11 records robustly with optional fast disk cache and stratified sampling.
    
    Returns:
        (records, statistics_dict)
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Webis dataset directory not found: {dataset_dir}")

    cache_dir = os.path.join("outputs", "cache")
    cache_path = os.path.join(cache_dir, "webis_dataset_cache.joblib")

    # Fast-path: load cached records if available and using the standard dataset directory
    if use_cache and os.path.normpath(dataset_dir) == os.path.normpath("data/Webis-CPC-11") and os.path.isfile(cache_path):
        try:
            cached_data = joblib.load(cache_path)
            records = cached_data["records"]
            stats = dict(cached_data["stats"])
            if max_records is not None and max_records < len(records):
                records = _sample_records(records, max_records, stratified=stratified, random_seed=random_seed)
                pos = sum(1 for r in records if r.label == 1)
                stats.update({
                    "total_processed": len(records),
                    "valid_records": len(records),
                    "positive_labels": pos,
                    "negative_labels": len(records) - pos
                })
            if verbose:
                print(f"[Webis-CPC-11] (Loaded from cache) Records: {stats['valid_records']} "
                      f"(Pos: {stats['positive_labels']}, Neg: {stats['negative_labels']})")
            return records, stats
        except Exception:
            pass  # Fall back to scanning if cache is corrupted

    # Discover record IDs and files with a single os.scandir pass (fast in-memory lookup)
    discovered_ids = set()
    available_files = set()
    with os.scandir(dataset_dir) as it:
        for entry in it:
            if entry.is_file():
                available_files.add(entry.name)
                if entry.name.endswith("-metadata.txt"):
                    rec_id = entry.name[:-len("-metadata.txt")]
                    discovered_ids.add(rec_id)

    # Sort record IDs (numerically if possible, else alphabetically)
    def _sort_key(s):
        try:
            return (0, int(s))
        except ValueError:
            return (1, s)

    sorted_ids = sorted(discovered_ids, key=_sort_key)
    target_ids = sorted_ids[:max_records] if max_records is not None else sorted_ids

    records: List[WebisRecord] = []
    missing_files = []
    invalid_labels = []
    skipped_records = []
    seen_ids = set()
    duplicate_count = 0

    pos_count = 0
    neg_count = 0

    for rec_id in target_ids:
        if rec_id in seen_ids:
            duplicate_count += 1
            skipped_records.append(rec_id)
            continue
        seen_ids.add(rec_id)

        meta_name = f"{rec_id}-metadata.txt"
        orig_name = f"{rec_id}-original.txt"
        para_name = f"{rec_id}-paraphrase.txt"

        # Check required files
        if not (orig_name in available_files and para_name in available_files and meta_name in available_files):
            missing_files.append(rec_id)
            skipped_records.append(rec_id)
            continue

        meta_path = os.path.join(dataset_dir, meta_name)
        orig_path = os.path.join(dataset_dir, orig_name)
        para_path = os.path.join(dataset_dir, para_name)

        meta = parse_metadata_file(meta_path)
        raw_label = meta.get("Paraphrase", "").strip().lower()
        if raw_label in ("yes", "1", "true"):
            label = 1
            pos_count += 1
        elif raw_label in ("no", "0", "false"):
            label = 0
            neg_count += 1
        else:
            invalid_labels.append((rec_id, meta.get("Paraphrase", "")))
            skipped_records.append(rec_id)
            continue

        try:
            with open(orig_path, "r", encoding="utf-8", errors="replace") as f:
                orig_text = f.read().strip()
            with open(para_path, "r", encoding="utf-8", errors="replace") as f:
                para_text = f.read().strip()
        except OSError:
            skipped_records.append(rec_id)
            continue

        records.append(WebisRecord(
            record_id=rec_id,
            original_text=orig_text,
            paraphrase_text=para_text,
            label=label,
            metadata=meta
        ))

    if max_records is not None and max_records < len(records):
        records = _sample_records(records, max_records, stratified=stratified, random_seed=random_seed)
        pos_count = sum(1 for r in records if r.label == 1)
        neg_count = len(records) - pos_count

    fingerprint = compute_dataset_fingerprint(records)

    stats = {
        "dataset_dir": dataset_dir,
        "dataset_name": "Webis-CPC-11",
        "dataset_fingerprint": fingerprint,
        "total_discovered": len(discovered_ids),
        "total_processed": len(target_ids),
        "valid_records": len(records),
        "positive_labels": pos_count,
        "negative_labels": neg_count,
        "missing_file_count": len(missing_files),
        "invalid_label_count": len(invalid_labels),
        "duplicate_count": duplicate_count,
        "skipped_count": len(skipped_records),
    }

    # Save cache if loaded the full standard dataset
    if max_records is None and os.path.normpath(dataset_dir) == os.path.normpath("data/Webis-CPC-11"):
        try:
            os.makedirs(cache_dir, exist_ok=True)
            joblib.dump({"records": records, "stats": stats}, cache_path, compress=3)
        except Exception:
            pass

    if verbose:
        print(f"[Webis-CPC-11] Discovered: {stats['total_discovered']}, Valid: {stats['valid_records']} "
              f"(Pos: {stats['positive_labels']}, Neg: {stats['negative_labels']}), "
              f"Skipped: {stats['skipped_count']}")

    return records, stats
