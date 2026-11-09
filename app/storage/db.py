"""Repository SQLite (stdlib sqlite3). Satu koneksi per operasi -> aman dipakai dari
event loop dan dari worker thread GPU sekaligus. WAL mode untuk concurrent read."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS doctors (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    anchor_path TEXT NOT NULL,
    enroll_seconds REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    doctor_id TEXT REFERENCES doctors(id),
    template_id TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    error TEXT,
    audio_path TEXT,
    extra_names TEXT NOT NULL DEFAULT '[]',
    locked_json TEXT NOT NULL DEFAULT '{}',
    client_ip TEXT,
    result_json TEXT,
    review_json TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
CREATE TABLE IF NOT EXISTS osce_cases (
    id TEXT PRIMARY KEY,
    job_id TEXT UNIQUE NOT NULL REFERENCES jobs(id),
    template_id TEXT NOT NULL,
    doctor_id TEXT,
    transcript_json TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Repository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------------- doctors ----------------
    def create_doctor(self, name: str, anchor_path: Path, enroll_seconds: float, doctor_id: str | None = None) -> dict:
        doctor_id = doctor_id or uuid.uuid4().hex[:12]
        with self._write_lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO doctors(id, name, anchor_path, enroll_seconds, created_at) VALUES (?,?,?,?,?)",
                (doctor_id, name, str(anchor_path), enroll_seconds, utcnow()),
            )
        return self.get_doctor(doctor_id)  # type: ignore[return-value]

    def get_doctor(self, doctor_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM doctors WHERE id=?", (doctor_id,)).fetchone()
        return dict(row) if row else None

    def list_doctors(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM doctors ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    # ---------------- jobs ----------------
    def create_job(self, job_id: str, doctor_id: str | None, template_id: str, audio_path: Path,
                   extra_names: list[str], locked: dict[str, Any], client_ip: str | None) -> dict:
        with self._write_lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO jobs(id, doctor_id, template_id, status, stage, audio_path, extra_names, locked_json,"
                " client_ip, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (job_id, doctor_id, template_id, "queued", "queued", str(audio_path),
                 json.dumps(extra_names, ensure_ascii=False), json.dumps(locked, ensure_ascii=False),
                 client_ip, utcnow()),
            )
        return self.get_job(job_id)  # type: ignore[return-value]

    def get_job(self, job_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        job = dict(row)
        job["extra_names"] = json.loads(job["extra_names"] or "[]")
        job["locked"] = json.loads(job.pop("locked_json") or "{}")
        for key in ("result_json", "review_json"):
            job[key.replace("_json", "")] = json.loads(job.pop(key)) if job.get(key) else None
        return job

    def list_jobs(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, doctor_id, template_id, status, stage, error, created_at, finished_at"
                " FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def jobs_with_status(self, statuses: tuple[str, ...]) -> list[dict]:
        placeholders = ",".join("?" * len(statuses))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT id, status FROM jobs WHERE status IN ({placeholders}) ORDER BY created_at", statuses
            ).fetchall()
        return [dict(r) for r in rows]

    def update_job(self, job_id: str, **fields: Any) -> None:
        allowed = {"status", "stage", "error", "audio_path", "started_at", "finished_at"}
        json_fields = {"result": "result_json", "review": "review_json"}
        sets: list[str] = []
        values: list[Any] = []
        for key, value in fields.items():
            if key in allowed:
                sets.append(f"{key}=?")
                values.append(value)
            elif key in json_fields:
                sets.append(f"{json_fields[key]}=?")
                values.append(None if value is None else json.dumps(value, ensure_ascii=False))
            else:
                raise ValueError(f"Kolom tidak diizinkan: {key}")
        if not sets:
            return
        values.append(job_id)
        with self._write_lock, self._conn() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", values)

    # ---------------- OSCE cases ----------------
    def create_case(self, job_id: str, template_id: str, doctor_id: str | None,
                    transcript: list[dict], summary: dict, metadata: dict) -> dict:
        case_id = uuid.uuid4().hex[:12]
        with self._write_lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO osce_cases(id, job_id, template_id, doctor_id, transcript_json, summary_json,"
                " metadata_json, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (case_id, job_id, template_id, doctor_id, json.dumps(transcript, ensure_ascii=False),
                 json.dumps(summary, ensure_ascii=False), json.dumps(metadata, ensure_ascii=False), utcnow()),
            )
            conn.execute("UPDATE jobs SET status='verified' WHERE id=?", (job_id,))
        return self.get_case(case_id)  # type: ignore[return-value]

    def get_case(self, case_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM osce_cases WHERE id=?", (case_id,)).fetchone()
        return self._case_row(row) if row else None

    def case_for_job(self, job_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM osce_cases WHERE job_id=?", (job_id,)).fetchone()
        return self._case_row(row) if row else None

    def list_cases(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM osce_cases ORDER BY created_at DESC").fetchall()
        return [self._case_row(r) for r in rows]

    @staticmethod
    def _case_row(row: sqlite3.Row) -> dict:
        case = dict(row)
        for key in ("transcript_json", "summary_json", "metadata_json"):
            case[key.replace("_json", "")] = json.loads(case.pop(key))
        return case
