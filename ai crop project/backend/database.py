"""
PlantCare AI - SQLite Database Interface
Manages scan history, diagnostic records, and aggregate statistics.
"""

import sqlite3
import os
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "plantcare.db")


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Creates and returns a connection to the SQLite database.
    Enables dictionary-like row access.
    """
    path = db_path or os.getenv("DATABASE_PATH", DEFAULT_DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initializes the database schema if tables do not exist.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    filename TEXT NOT NULL,
                    plant_type TEXT NOT NULL,
                    disease_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    raw_class TEXT NOT NULL,
                    is_low_confidence INTEGER DEFAULT 0,
                    notes TEXT DEFAULT ''
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scans_timestamp 
                ON scans (timestamp DESC);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scans_status 
                ON scans (status);
            """)
    finally:
        conn.close()


def add_scan(
    plant_type: str,
    disease_name: str,
    status: str,
    confidence: float,
    raw_class: str,
    filename: str = "upload.jpg",
    is_low_confidence: bool = False,
    notes: str = "",
    db_path: Optional[str] = None
) -> int:
    """
    Inserts a newly processed leaf scan record into the database.
    Returns the new record ID.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO scans (
                    filename, plant_type, disease_name, status, 
                    confidence, raw_class, is_low_confidence, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    filename,
                    plant_type,
                    disease_name,
                    status,
                    round(float(confidence), 4),
                    raw_class,
                    1 if is_low_confidence else 0,
                    notes
                )
            )
            return cursor.lastrowid
    finally:
        conn.close()


def get_recent_scans(limit: int = 25, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves the most recent scans ordered by timestamp descending.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, timestamp, filename, plant_type, disease_name, 
                   status, confidence, raw_class, is_low_confidence, notes
            FROM scans
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_statistics(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes aggregate metrics from recorded scans:
    total scans, healthy vs diseased counts, average confidence,
    and frequency breakdown by disease.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        
        # Overall counts
        cursor.execute("SELECT COUNT(*) AS total FROM scans")
        total = cursor.fetchone()["total"]

        if total == 0:
            return {
                "total_scans": 0,
                "healthy_count": 0,
                "diseased_count": 0,
                "uncertain_count": 0,
                "average_confidence": 0.0,
                "disease_distribution": []
            }

        cursor.execute(
            "SELECT COUNT(*) AS c FROM scans WHERE LOWER(status) = 'healthy'"
        )
        healthy_count = cursor.fetchone()["c"]

        cursor.execute(
            "SELECT COUNT(*) AS c FROM scans WHERE LOWER(status) = 'diseased' AND is_low_confidence = 0"
        )
        diseased_count = cursor.fetchone()["c"]

        cursor.execute(
            "SELECT COUNT(*) AS c FROM scans WHERE is_low_confidence = 1 OR LOWER(status) = 'uncertain'"
        )
        uncertain_count = cursor.fetchone()["c"]

        cursor.execute("SELECT AVG(confidence) AS avg_conf FROM scans")
        avg_conf_row = cursor.fetchone()["avg_conf"]
        avg_confidence = round(float(avg_conf_row or 0.0) * 100, 1)

        # Distribution of top detected diseases
        cursor.execute(
            """
            SELECT disease_name, plant_type, COUNT(*) as count
            FROM scans
            WHERE LOWER(status) != 'healthy'
            GROUP BY disease_name, plant_type
            ORDER BY count DESC
            LIMIT 5
            """
        )
        distribution = [dict(row) for row in cursor.fetchall()]

        return {
            "total_scans": total,
            "healthy_count": healthy_count,
            "diseased_count": diseased_count,
            "uncertain_count": uncertain_count,
            "average_confidence": avg_confidence,
            "disease_distribution": distribution
        }
    finally:
        conn.close()


def clear_history(db_path: Optional[str] = None) -> int:
    """
    Deletes all scan records from the database.
    Returns the number of deleted records.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("DELETE FROM scans")
            return cursor.rowcount
    finally:
        conn.close()


# Ensure database is initialized when module is imported
try:
    init_db()
except Exception:
    pass
