"""상태 저장 — SQLite 한 파일 (작업·문장·화자 이름·A2A 작업). 서비스만 쓴다."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, kind TEXT NOT NULL, origin TEXT NOT NULL, caller TEXT,
  status TEXT NOT NULL, stage TEXT, audio_path TEXT, audio_name TEXT, duration REAL,
  lang TEXT, context TEXT, num_speakers INTEGER, speakers_on INTEGER NOT NULL DEFAULT 1, source_url TEXT,
  engine TEXT, model TEXT, diarizer TEXT, language TEXT, warning TEXT, error TEXT, error_kind TEXT,
  timings TEXT, created TEXT NOT NULL, updated TEXT NOT NULL, started TEXT, finished TEXT
);
CREATE TABLE IF NOT EXISTS segments (
  job_id TEXT NOT NULL, idx INTEGER NOT NULL, start REAL NOT NULL, end REAL NOT NULL, speaker TEXT,
  text TEXT NOT NULL, asr_text TEXT NOT NULL, timing TEXT, checked INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (job_id, idx)
);
CREATE TABLE IF NOT EXISTS speakers (job_id TEXT NOT NULL, label TEXT NOT NULL, name TEXT NOT NULL, PRIMARY KEY (job_id, label));
CREATE TABLE IF NOT EXISTS a2a_tasks (
  id TEXT PRIMARY KEY, context_id TEXT NOT NULL, message_id TEXT UNIQUE, job_id TEXT, state TEXT, error TEXT, created TEXT NOT NULL
);
"""

# 처음 스키마 뒤에 더한 칸 — 새 DB 든 옛 DB 든 시작할 때 없는 칸만 ALTER TABLE 로 더한다 (NOTES D12).
#   jobs: 출처 키(같은 영상 찾기) + 영상 정보 (yt-dlp -J 에서). 길이는 기존 duration 칸.
#   a2a_tasks: 이 A2A 작업이 기존 작업을 다시 쓴 것인지.
ADDED = {
    "jobs": {"source_key": "TEXT", "video_id": "TEXT", "channel": "TEXT", "upload_date": "TEXT", "site": "TEXT",
             "webpage_url": "TEXT"},
    "a2a_tasks": {"reused": "INTEGER NOT NULL DEFAULT 0"},
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        for table, cols in ADDED.items():
            have = {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})")}
            for name, decl in cols.items():
                if name not in have:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        self.db.execute("CREATE INDEX IF NOT EXISTS jobs_source_key ON jobs(source_key)")
        self.lock = threading.RLock()

    def _q(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def _x(self, sql: str, args=()) -> None:
        with self.lock:
            self.db.execute(sql, args)

    # ---------------- 작업 ----------------
    def add_job(self, **f) -> None:
        f.setdefault("created", now())
        f.setdefault("updated", f["created"])
        keys = ",".join(f)
        self._x(f"INSERT INTO jobs ({keys}) VALUES ({','.join('?' * len(f))})", tuple(f.values()))

    def update_job(self, jid: str, **f) -> None:
        if not f:
            return
        f["updated"] = now()
        if "timings" in f and isinstance(f["timings"], dict):
            f["timings"] = json.dumps(f["timings"])
        self._x(f"UPDATE jobs SET {','.join(k + '=?' for k in f)} WHERE id=?", (*f.values(), jid))

    def job(self, jid: str) -> Optional[dict]:
        r = self._q("SELECT * FROM jobs WHERE id=?", (jid,))
        return r[0] if r else None

    def jobs(self) -> list[dict]:
        """목록용 — 문장 수·확인 수·고침 수를 함께."""
        return self._q("""SELECT j.*, COUNT(s.idx) AS n_segments, COALESCE(SUM(s.checked),0) AS n_checked,
                          COALESCE(SUM(s.text != s.asr_text),0) AS n_edited,
                          (SELECT COUNT(DISTINCT speaker) FROM segments x WHERE x.job_id=j.id AND x.speaker IS NOT NULL) AS n_speakers
                          FROM jobs j LEFT JOIN segments s ON s.job_id=j.id GROUP BY j.id ORDER BY j.created DESC""")

    def jobs_with_status(self, *status: str) -> list[dict]:
        return self._q(f"SELECT * FROM jobs WHERE status IN ({','.join('?' * len(status))}) ORDER BY created",
                       status)

    def jobs_by_source(self, key: str) -> list[dict]:
        """같은 출처 키의 주소 작업 — 최근 것부터."""
        return self._q("SELECT * FROM jobs WHERE source_key=? AND kind='file' ORDER BY created DESC, rowid DESC", (key,))

    def backfill_source_keys(self, key_of) -> None:
        """출처 키가 비어 있는 주소 작업에 key_of(주소) 를 채운다 — updated 는 건드리지 않는다 (사람이 고친 시각이 아님)."""
        with self.lock:
            rows = self.db.execute("SELECT id, source_url FROM jobs WHERE kind='file' AND source_url IS NOT NULL "
                                   "AND source_key IS NULL").fetchall()
            for r in rows:
                self.db.execute("UPDATE jobs SET source_key=? WHERE id=?", (key_of(r["source_url"]), r["id"]))

    def delete_job(self, jid: str) -> None:
        with self.lock:
            for t in ("segments", "speakers"):
                self.db.execute(f"DELETE FROM {t} WHERE job_id=?", (jid,))
            self.db.execute("DELETE FROM jobs WHERE id=?", (jid,))

    # ---------------- 문장 ----------------
    def replace_segments(self, jid: str, segs: list[dict]) -> None:
        with self.lock:
            self.db.execute("BEGIN")
            self.db.execute("DELETE FROM segments WHERE job_id=?", (jid,))
            self.db.executemany(
                "INSERT INTO segments (job_id, idx, start, end, speaker, text, asr_text, timing) VALUES (?,?,?,?,?,?,?,?)",
                [(jid, i, s["start"], s["end"], s.get("speaker"), s["text"], s["text"], s.get("timing")) for i, s in enumerate(segs)])
            self.db.execute("COMMIT")

    def append_segment(self, jid: str, s: dict) -> int:
        with self.lock:
            n = self.db.execute("SELECT COALESCE(MAX(idx)+1,0) FROM segments WHERE job_id=?", (jid,)).fetchone()[0]
            self.db.execute("INSERT INTO segments (job_id, idx, start, end, speaker, text, asr_text, timing) VALUES (?,?,?,?,?,?,?,?)",
                            (jid, n, s["start"], s["end"], s.get("speaker"), s["text"], s["text"], s.get("timing")))
            return n

    def segments(self, jid: str) -> list[dict]:
        return self._q("SELECT idx, start, end, speaker, text, asr_text, timing, checked FROM segments WHERE job_id=? ORDER BY idx", (jid,))

    def segment(self, jid: str, idx: int) -> Optional[dict]:
        r = self._q("SELECT idx, start, end, speaker, text, asr_text, timing, checked FROM segments WHERE job_id=? AND idx=?", (jid, idx))
        return r[0] if r else None

    def update_segment(self, jid: str, idx: int, **f) -> None:
        self._x(f"UPDATE segments SET {','.join(k + '=?' for k in f)} WHERE job_id=? AND idx=?", (*f.values(), jid, idx))

    def check_until(self, jid: str, idx: int) -> None:
        self._x("UPDATE segments SET checked=1 WHERE job_id=? AND idx<=?", (jid, idx))

    def search(self, q: str, limit: int = 50) -> list[dict]:
        """제목·채널이 걸린 작업(idx 없음) 먼저, 그다음 글이 걸린 문장."""
        like = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        jobs = self._q("""SELECT id AS job_id, NULL AS idx, NULL AS start, NULL AS speaker, NULL AS text, title, channel FROM jobs
                          WHERE title LIKE ? ESCAPE '\\' OR channel LIKE ? ESCAPE '\\' ORDER BY created DESC LIMIT ?""",
                       (like, like, limit))
        return jobs + self._q("""SELECT s.job_id, s.idx, s.start, s.speaker, s.text, j.title, j.channel FROM segments s
                          JOIN jobs j ON j.id=s.job_id WHERE s.text LIKE ? ESCAPE '\\' ORDER BY j.created DESC, s.idx LIMIT ?""",
                              (like, limit))

    # ---------------- 화자 이름 ----------------
    def speaker_names(self, jid: str) -> dict:
        return {r["label"]: r["name"] for r in self._q("SELECT label, name FROM speakers WHERE job_id=?", (jid,))}

    def set_speaker_name(self, jid: str, label: str, name: str) -> None:
        if name:
            self._x("INSERT INTO speakers (job_id, label, name) VALUES (?,?,?) ON CONFLICT(job_id,label) DO UPDATE SET name=excluded.name",
                    (jid, label, name))
        else:
            self._x("DELETE FROM speakers WHERE job_id=? AND label=?", (jid, label))

    def merge_speaker(self, jid: str, label: str, into: str) -> int:
        with self.lock:
            n = self.db.execute("UPDATE segments SET speaker=? WHERE job_id=? AND speaker=?", (into, jid, label)).rowcount
            self.db.execute("DELETE FROM speakers WHERE job_id=? AND label=?", (jid, label))
            return n

    def speaker_labels(self, jid: str) -> set:
        return {r["speaker"] for r in self._q("SELECT DISTINCT speaker FROM segments WHERE job_id=? AND speaker IS NOT NULL", (jid,))}

    def clear_speaker_names(self, jid: str) -> None:
        self._x("DELETE FROM speakers WHERE job_id=?", (jid,))

    # ---------------- A2A ----------------
    def add_task(self, tid: str, context_id: str, message_id: Optional[str], job_id: Optional[str], state: Optional[str] = None,
                 error: Optional[str] = None, reused: bool = False) -> None:
        self._x("INSERT INTO a2a_tasks (id, context_id, message_id, job_id, state, error, created, reused) VALUES (?,?,?,?,?,?,?,?)",
                (tid, context_id, message_id, job_id, state, error, now(), int(reused)))

    def task(self, tid: str) -> Optional[dict]:
        r = self._q("SELECT * FROM a2a_tasks WHERE id=?", (tid,))
        return r[0] if r else None

    def task_by_message(self, mid: str) -> Optional[dict]:
        r = self._q("SELECT * FROM a2a_tasks WHERE message_id=?", (mid,))
        return r[0] if r else None

    def set_task_state(self, tid: str, state: str) -> None:
        self._x("UPDATE a2a_tasks SET state=? WHERE id=?", (state, tid))

    def tasks_of_job(self, job_id: str) -> list[dict]:
        return self._q("SELECT * FROM a2a_tasks WHERE job_id=?", (job_id,))

    def tasks(self, context_id: Optional[str] = None) -> list[dict]:
        if context_id:
            return self._q("SELECT * FROM a2a_tasks WHERE context_id=? ORDER BY created DESC", (context_id,))
        return self._q("SELECT * FROM a2a_tasks ORDER BY created DESC")
