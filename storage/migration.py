"""
SQLite to MongoDB Migration Utility.

Migrates existing scan history from SQLite (plagiarism_history.db) into MongoDB,
preserving all original IDs, timestamps, statistics, and detailed sentence/source comparisons.
Validates migration counts and generates a verification report without altering the SQLite source.
"""
import argparse
import json
import os
import sqlite3
import sys
from typing import Any, Dict, Optional

from storage.sqlite_backend import SQLiteStorageBackend


def migrate_sqlite_to_mongodb(
    sqlite_path: str = "plagiarism_history.db",
    mongo_uri: Optional[str] = None,
    mongo_db_name: Optional[str] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Execute migration from SQLite to MongoDB.
    """
    if not os.path.isfile(sqlite_path):
        raise FileNotFoundError(f"Source SQLite database not found at {sqlite_path}")

    from storage.mongo_backend import MongoStorageBackend
    mongo = MongoStorageBackend(uri=mongo_uri, db_name=mongo_db_name)

    # Read SQLite records
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scans ORDER BY id ASC")
    rows = cursor.fetchall()

    total_sqlite_rows = len(rows)
    migrated_count = 0
    duplicate_count = 0
    error_count = 0
    errors = []

    print(f"\n[Migration] Found {total_sqlite_rows} scans in SQLite '{sqlite_path}'.")

    for row in rows:
        r_dict = dict(row)
        scan_id = r_dict["id"]

        # Check for duplicates in MongoDB
        existing = mongo.scan_history.find_one({"scan_id": scan_id})
        if existing:
            duplicate_count += 1
            continue

        details_str = r_dict.get("details")
        details = {}
        if details_str:
            try:
                details = json.loads(details_str)
            except Exception as e:
                errors.append(f"Scan {scan_id} JSON parse error: {e}")

        # Ensure ID in details matches
        details["id"] = scan_id
        details["scanned_at"] = r_dict["scan_date"]
        details["document_name"] = r_dict["document_name"]
        details["status"] = r_dict["status"]
        if "stats" not in details:
            details["stats"] = {
                "total_sentences": r_dict["total_sentences"],
                "exact": r_dict["exact_matches"],
                "near_exact": r_dict["near_exact_matches"],
                "semantic": r_dict["semantic_matches"],
                "verified_sources": r_dict["verified_sources"],
                "sources_with_matches": r_dict["sources_with_matches"],
                "overall_pct": r_dict["overall_similarity"],
                "exact_pct": r_dict["exact_pct"],
                "semantic_pct": r_dict["semantic_pct"],
            }

        if not dry_run:
            try:
                mongo.save_scan(details)
                migrated_count += 1
            except Exception as e:
                error_count += 1
                errors.append(f"Scan {scan_id} insert failed: {e}")
        else:
            migrated_count += 1

    conn.close()

    # Verification: check total count in MongoDB
    mongo_total = mongo.scan_history.count_documents({})

    report = {
        "sqlite_source_path": sqlite_path,
        "total_sqlite_scans": total_sqlite_rows,
        "newly_migrated": migrated_count,
        "duplicates_skipped": duplicate_count,
        "errors_count": error_count,
        "errors": errors,
        "total_mongodb_scans_now": mongo_total,
        "dry_run": dry_run,
        "status": "SUCCESS" if error_count == 0 else "COMPLETED_WITH_ERRORS"
    }

    print("\n" + "=" * 60)
    print("MIGRATION REPORT")
    print("=" * 60)
    print(f"Status: {report['status']}")
    print(f"SQLite Records: {total_sqlite_rows}")
    print(f"Newly Migrated: {migrated_count}")
    print(f"Duplicates Skipped: {duplicate_count}")
    print(f"Errors: {error_count}")
    print(f"MongoDB Total Records: {mongo_total}")
    print("=" * 60 + "\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="Migrate scan history from SQLite to MongoDB.")
    parser.add_argument("--sqlite-path", default="plagiarism_history.db", help="Path to SQLite database.")
    parser.add_argument("--mongo-uri", default=None, help="MongoDB connection URI.")
    parser.add_argument("--db-name", default=None, help="Target MongoDB database name.")
    parser.add_argument("--dry-run", action="store_true", help="Perform verification without writing to MongoDB.")
    args = parser.parse_args()

    try:
        migrate_sqlite_to_mongodb(
            sqlite_path=args.sqlite_path,
            mongo_uri=args.mongo_uri,
            mongo_db_name=args.db_name,
            dry_run=args.dry_run
        )
    except Exception as e:
        print(f"[Migration Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
