"""서비스 — 상태의 유일한 주인. 제품 API(app.py)와 A2A(a2a_tasks.py)가 같은 함수를 부른다.

작업 하나 = 받아 적을 녹음 하나 (파일 또는 실시간). GPU 를 쓰는 일은 일꾼 스레드 하나가 차례로 (받아 적기·구간 다시).
실시간은 따로 띄운 엔진 프로세스가 맡고, 끝나면 녹음본을 같은 일꾼 줄에 넣어 최종본을 만든다.
재시작하면 줄에 있던 일(queued·running)은 처음부터 다시, 듣던 실시간은 녹음된 만큼 최종본으로.
"""
from __future__ import annotations

import json
import queue
import shutil
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Optional, Protocol

from stt import formats
from stt.engine import EngineError, Selection
from stt.store import Store, now

STAGES = {"fetch": "소리 받는 중", "prepare": "소리 꺼내는 중", "diarize": "화자 나누는 중", "transcribe": "받아 적는 중",
          "final": "최종본 만드는 중", "load": "엔진 준비 중", "listen": "듣는 중"}
ENABLE_HINT = ("받아 적기 엔진을 쓸 수 없습니다. models.json 의 python 환경(voice-eval)과 모델이 있는지, GPU 여유가 있는지 확인하세요 "
               "— `python -m stt transcribe --dry-run <파일>` 이 이유를 보여 줍니다.")


class EngineLike(Protocol):
    """서비스가 쓰는 엔진 모양 — 테스트는 가짜를 넣는다 (모델만 가짜)."""
    def select(self, lang: Optional[str] = None, engine: Optional[str] = None, speakers: bool = True) -> Selection: ...
    def model_of(self, name: str) -> str: ...
    def prepare(self, src: Path, work: Path, *, start: Optional[float] = None, end: Optional[float] = None,
                name: str = "audio16k.wav") -> tuple[Path, float]: ...
    def diarize(self, wav: Path, work: Path, num_speakers: Optional[int] = None, cancel=None) -> list[dict]: ...
    def transcribe(self, name: str, wav: Path, pieces=None, *, lang=None, context: str = "", cancel=None) -> dict: ...
    def live(self, name: str, *, lang=None, context: str = ""): ...
    def fetch(self, url: str, dest_dir: Path, *, cancel=None) -> tuple[Path, dict]: ...


class ServiceError(Exception):
    def __init__(self, message: str, http: int = 400, kind: Optional[str] = None):
        super().__init__(message)
        self.message, self.http, self.kind = message, http, kind


class Service:
    def __init__(self, data_dir: Path, engine: EngineLike, *, base_url: str = "http://127.0.0.1:8773", start_worker: bool = True):
        self.data = Path(data_dir)
        self.audio_dir = self.data / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.data / "stt.db")
        self.engine = engine
        self.base_url = base_url
        self.q: queue.Queue = queue.Queue()
        self.cancels: dict[str, threading.Event] = {}
        self.redoing: set[tuple[str, int]] = set()
        self.lives: dict[str, "LiveRun"] = {}
        self.cond = threading.Condition()
        self.version = 0
        self.changed: list[tuple[int, str]] = []
        self._status_cache: tuple[float, dict] = (0.0, {})
        self._resume()
        if start_worker:
            threading.Thread(target=self._worker, daemon=True, name="stt-worker").start()

    # ---------------- 알림 (SSE) ----------------
    def notify(self, jid: str = "") -> None:
        with self.cond:
            self.version += 1
            self.changed = (self.changed + [(self.version, jid)])[-500:]
            self.cond.notify_all()

    def wait_change(self, since: int, timeout: float = 15.0) -> tuple[int, list[str]]:
        with self.cond:
            if self.version <= since:
                self.cond.wait(timeout)
            ids = sorted({j for v, j in self.changed if v > since and j})
            return self.version, ids

    # ---------------- 상태 ----------------
    def status(self, fresh: bool = False) -> dict:
        t, cached = self._status_cache
        if cached and not fresh and time.time() - t < 10:
            return cached
        sel = self.engine.select()
        st = {"asr": sel.asr, "diarizer": sel.diarizer, "selection": sel.trail, "ok": bool(sel.asr),
              "hint": None if sel.asr else ENABLE_HINT, "queue": self.q.qsize(), "live": list(self.lives)}
        self._status_cache = (time.time(), st)
        return st

    # ---------------- 작업 만들기 ----------------
    def create_file_job(self, src: Path, name: str, *, title: Optional[str] = None, lang: Optional[str] = None,
                        context: str = "", num_speakers: Optional[int] = None, speakers: bool = True,
                        origin: str = "screen", caller: Optional[str] = None, move: bool = False) -> dict:
        if not src.is_file() or src.stat().st_size == 0:
            raise ServiceError("빈 파일이거나 파일이 없습니다")
        jid = uuid.uuid4().hex[:12]
        dst = self.audio_dir / f"{jid}{Path(name).suffix.lower() or '.bin'}"
        (shutil.move if move else shutil.copyfile)(str(src), str(dst))
        lang = (lang or "").strip().lower() or None
        self.store.add_job(id=jid, title=(title or Path(name).stem)[:200], kind="file", origin=origin, caller=caller,
                           status="queued", stage=None, audio_path=str(dst), audio_name=name, lang=lang,
                           context=(context or "")[:2000], num_speakers=num_speakers, speakers_on=int(speakers))
        self.q.put(("job", jid))
        self.notify(jid)
        return self.job_summary(jid)

    def create_url_job(self, url: str, *, title: Optional[str] = None, lang: Optional[str] = None, context: str = "",
                       num_speakers: Optional[int] = None, speakers: bool = True, origin: str = "screen",
                       caller: Optional[str] = None) -> dict:
        """주소(유튜브 등)로 받아 적기 — 소리는 일꾼이 받는다 (줄·취소·재시작이 받아 적기와 같은 흐름)."""
        url = (url or "").strip()
        if not url.startswith(("http://", "https://")) or len(url) > 2000:
            raise ServiceError("http(s) 주소를 넣어 주세요")
        jid = uuid.uuid4().hex[:12]
        lang = (lang or "").strip().lower() or None
        self.store.add_job(id=jid, title=(title or url)[:200], kind="file", origin=origin, caller=caller, status="queued",
                           stage=None, audio_path=None, audio_name=None, lang=lang, context=(context or "")[:2000],
                           num_speakers=num_speakers, speakers_on=int(speakers), source_url=url,
                           warning=None if title else "제목은 주소를 읽으면 채웁니다")
        self.q.put(("job", jid))
        self.notify(jid)
        return self.job_summary(jid)

    def _fetch(self, j: dict, cancel: threading.Event) -> dict:
        """주소 작업의 소리를 받아 붙인다. 제목이 주소 그대로면 영상 제목으로, 맥락이 비었으면 제목·채널·설명으로."""
        self._stage(j["id"], "fetch")
        with tempfile.TemporaryDirectory(prefix="stt-fetch-") as td:
            got, info = self.engine.fetch(j["source_url"], Path(td), cancel=cancel)
            dst = self.audio_dir / f"{j['id']}{got.suffix.lower()}"
            shutil.move(str(got), str(dst))
        f: dict = {"audio_path": str(dst), "audio_name": None, "warning": None}
        if info.get("title") and j["title"] == j["source_url"]:
            f["title"] = str(info["title"])[:200]
        if not (j["context"] or "").strip():
            parts = [info.get("title"), info.get("uploader"), (info.get("description") or "")[:800]]
            f["context"] = "\n".join(str(x) for x in parts if x)[:2000]
        if info.get("duration"):
            f["duration"] = float(info["duration"])
        self.store.update_job(j["id"], **f)
        return self.store.job(j["id"])

    # ---------------- 일꾼 ----------------
    def _resume(self) -> None:
        for j in self.store.jobs_with_status("queued", "running"):
            self.store.update_job(j["id"], status="queued", stage=None)
            self.q.put(("job", j["id"]))
        for j in self.store.jobs_with_status("live"):     # 듣던 중 서버가 꺼짐 → 녹음된 만큼 최종본으로
            p = Path(j["audio_path"] or "")
            if p.exists() and p.stat().st_size > 44 + 32000:
                self.store.update_job(j["id"], status="queued", stage=None, warning="서버가 다시 켜져 녹음된 만큼으로 최종본을 만듦")
                self.q.put(("job", j["id"]))
            else:
                self.store.update_job(j["id"], status="failed", error="듣던 중 서버가 꺼졌고 녹음된 소리가 없습니다", error_kind="bad_audio")

    def _worker(self) -> None:
        while True:
            item = self.q.get()
            try:
                if item[0] == "job":
                    self._run_job(item[1])
                elif item[0] == "redo":
                    self._run_redo(item[1], item[2])
            except Exception as ex:  # noqa: BLE001 — 일꾼은 죽지 않는다. 작업에 남긴다
                if item[0] == "job":
                    self.store.update_job(item[1], status="failed", error=f"내부 오류: {type(ex).__name__}: {ex}"[:500],
                                          error_kind="failed")
                    self.notify(item[1])
            finally:
                self.q.task_done()

    def _stage(self, jid: str, key: str) -> None:
        self.store.update_job(jid, stage=STAGES[key])
        self.notify(jid)

    def _run_job(self, jid: str) -> None:
        j = self.store.job(jid)
        if not j or j["status"] != "queued":
            return                                         # 취소·삭제된 일
        cancel = self.cancels.setdefault(jid, threading.Event())
        final = j["kind"] == "live"
        self.store.update_job(jid, status="running", started=now(), error=None, error_kind=None)
        t0, timings = time.perf_counter(), {}
        work = Path(tempfile.mkdtemp(prefix="stt-job-"))
        try:
            sel = self.engine.select(j["lang"], speakers=bool(j["speakers_on"]))
            if not sel.asr:
                raise EngineError("no_engine", ENABLE_HINT)
            if j["source_url"] and not final and not (j["audio_path"] and Path(j["audio_path"]).exists()):
                j = self._fetch(j, cancel)                 # 주소 작업: 소리부터 (재시작하면 받은 것이 없을 때만 다시)
            self._stage(jid, "final" if final else "prepare")
            wav, dur = self.engine.prepare(Path(j["audio_path"]), work)
            self.store.update_job(jid, duration=round(dur, 2))
            timings["prepare_s"] = round(time.perf_counter() - t0, 1)
            pieces, warning = None, None
            if sel.diarizer:
                if not final:
                    self._stage(jid, "diarize")
                t = time.perf_counter()
                try:
                    pieces = self.engine.diarize(wav, work, j["num_speakers"], cancel)
                except EngineError as ex:
                    if ex.kind == "canceled":
                        raise
                    warning = ("말소리를 찾지 못해 화자 없이 받아 적었습니다" if ex.kind == "no_speech"
                               else f"화자 분리 실패 — 화자 없이 받아 적었습니다 ({ex.message[-120:]})")
                timings["diarize_s"] = round(time.perf_counter() - t, 1)
            elif j["speakers_on"]:
                warning = "화자 분리 엔진을 쓸 수 없어 화자 없이 받아 적었습니다"
            if not final:
                self._stage(jid, "transcribe")
            res, used, errors = None, None, []
            for name in sel.asr:                           # 실행 중 실패(메모리 부족 등)면 다음 후보 — 후보 수가 상한
                try:
                    res, used = self.engine.transcribe(name, wav, pieces, lang=j["lang"], context=j["context"] or "",
                                                       cancel=cancel), name
                    break
                except EngineError as ex:
                    if ex.kind == "canceled":
                        raise
                    errors.append(ex.message[-200:])
            if res is None:
                raise EngineError("failed", "모든 받아 적기 엔진이 실패했습니다: " + " / ".join(errors))
            timings.update(load_s=res.get("load_s"), run_s=res.get("run_s"), total_s=round(time.perf_counter() - t0, 1))
            if cancel.is_set():
                raise EngineError("canceled", "취소함")
            self.store.replace_segments(jid, res["segments"])
            self.store.clear_speaker_names(jid)
            self.store.update_job(jid, status="done", stage=None, engine=used, model=self.engine.model_of(used),
                                  diarizer=sel.diarizer if pieces is not None else None, language=res.get("language"),
                                  warning=warning or j.get("warning"), timings=timings, finished=now())
        except EngineError as ex:
            if ex.kind == "canceled":
                self.store.update_job(jid, status="canceled", stage=None, finished=now())
            else:
                self.store.update_job(jid, status="failed", stage=None, error=ex.message, error_kind=ex.kind, finished=now())
        finally:
            shutil.rmtree(work, ignore_errors=True)
            self.cancels.pop(jid, None)
            self.notify(jid)

    def _run_redo(self, jid: str, idx: int) -> None:
        j, s = self.store.job(jid), self.store.segment(jid, idx)
        try:
            if not j or not s:
                return
            sel = self.engine.select(j["lang"], speakers=False)
            if not sel.asr:
                raise EngineError("no_engine", ENABLE_HINT)
            with tempfile.TemporaryDirectory(prefix="stt-redo-") as td:
                wav, _ = self.engine.prepare(Path(j["audio_path"]), Path(td), start=max(0.0, s["start"] - 0.15), end=s["end"] + 0.15)
                res = self.engine.transcribe(sel.asr[0], wav, None, lang=j["lang"], context=j["context"] or "")
            text = " ".join(x["text"] for x in res["segments"]).strip()
            if text:
                self.store.update_segment(jid, idx, text=text, asr_text=text, checked=0)
        except EngineError as ex:
            self.store.update_job(jid, warning=f"{idx + 1}번 문장 다시 받아 적기 실패: {ex.message[-160:]}")
        finally:
            self.redoing.discard((jid, idx))
            self.notify(jid)

    # ---------------- 사람 동작 ----------------
    def _job(self, jid: str) -> dict:
        j = self.store.job(jid)
        if not j:
            raise ServiceError("작업이 없습니다", 404)
        return j

    def _seg(self, jid: str, idx: int) -> dict:
        s = self.store.segment(jid, idx)
        if not s:
            raise ServiceError("문장이 없습니다", 404)
        return s

    def cancel(self, jid: str) -> dict:
        j = self._job(jid)
        if j["status"] == "queued":
            self.store.update_job(jid, status="canceled", finished=now())
        elif j["status"] == "running":
            self.cancels.setdefault(jid, threading.Event()).set()
        elif j["status"] == "live":
            self.stop_live(jid, final=False)
        else:
            raise ServiceError("끝난 작업은 취소할 수 없습니다", 409, "not_cancelable")
        self.notify(jid)
        return self.job_summary(jid)

    def retranscribe(self, jid: str) -> dict:
        j = self._job(jid)
        if j["status"] in ("queued", "running", "live"):
            raise ServiceError("받아 적는 중인 작업입니다", 409)
        self.store.update_job(jid, status="queued", stage=None, error=None, error_kind=None, warning=None)
        self.q.put(("job", jid))
        self.notify(jid)
        return self.job_summary(jid)

    def delete(self, jid: str) -> None:
        j = self._job(jid)
        if j["status"] in ("running", "live"):
            raise ServiceError("받아 적는 중에는 지울 수 없습니다 — 먼저 취소하세요", 409)
        self.store.delete_job(jid)
        if j["audio_path"]:
            Path(j["audio_path"]).unlink(missing_ok=True)
        self.notify(jid)


    def rename(self, jid: str, title: str) -> dict:
        self._job(jid)
        title = title.strip()
        if not title:
            raise ServiceError("제목이 비었습니다")
        self.store.update_job(jid, title=title[:200])
        self.notify(jid)
        return self.job_summary(jid)

    def edit_segment(self, jid: str, idx: int, *, text: Optional[str] = None, checked: Optional[bool] = None) -> dict:
        self._job(jid)
        s = self._seg(jid, idx)
        f: dict = {}
        if text is not None:
            text = " ".join(text.split())
            if not text:
                raise ServiceError("문장이 비었습니다 — 지우려면 원래 글로 되돌리세요")
            f.update(text=text, checked=1)                 # 고친 문장은 확인한 것으로
        if checked is not None:
            f["checked"] = int(checked)
        if f:
            self.store.update_segment(jid, idx, **f)
            self.notify(jid)
        return self.store.segment(jid, idx) or s

    def revert_segment(self, jid: str, idx: int) -> dict:
        s = self._seg(jid, idx)
        self.store.update_segment(jid, idx, text=s["asr_text"])
        self.notify(jid)
        return self.store.segment(jid, idx)

    def check_until(self, jid: str, idx: int) -> dict:
        self._seg(jid, idx)
        self.store.check_until(jid, idx)
        self.notify(jid)
        return self.job_summary(jid)

    def redo_segment(self, jid: str, idx: int) -> dict:
        j = self._job(jid)
        self._seg(jid, idx)
        if j["status"] != "done":
            raise ServiceError("받아 적기가 끝난 작업에서만 됩니다", 409)
        if (jid, idx) not in self.redoing:
            self.redoing.add((jid, idx))
            self.q.put(("redo", jid, idx))
            self.notify(jid)
        return {"job_id": jid, "idx": idx, "redoing": True}

    def rename_speaker(self, jid: str, label: str, name: str) -> dict:
        self._job(jid)
        self.store.set_speaker_name(jid, label, " ".join(name.split())[:60])
        self.notify(jid)
        return self.store.speaker_names(jid)

    def merge_speaker(self, jid: str, label: str, into: str) -> dict:
        """화자 분리가 한 사람을 둘로 나눴을 때: label 의 문장을 모두 into 로 옮긴다 (이름은 into 쪽을 남김)."""
        j = self._job(jid)
        if j["status"] != "done":
            raise ServiceError("받아 적기가 끝난 작업에서만 됩니다", 409)
        labels = self.store.speaker_labels(jid)
        if label == into or label not in labels or into not in labels:
            raise ServiceError("합칠 두 화자를 이 작업 안에서 고르세요")
        n = self.store.merge_speaker(jid, label, into)
        self.notify(jid)
        return {"moved": n, "speakers": self.store.speaker_names(jid)}

    # ---------------- 읽기 ----------------
    def job_summary(self, jid: str) -> dict:
        for j in self.store.jobs():
            if j["id"] == jid:
                return self._summary(j)
        raise ServiceError("작업이 없습니다", 404)

    def _summary(self, j: dict) -> dict:
        out = {k: j[k] for k in ("id", "title", "kind", "origin", "caller", "status", "stage", "audio_name", "duration", "lang",
                                 "language", "engine", "model", "diarizer", "warning", "error", "error_kind", "created",
                                 "updated", "started", "finished", "n_segments", "n_checked", "n_edited", "n_speakers",
                                 "source_url", "num_speakers", "speakers_on", "context")}
        out["timings"] = json.loads(j["timings"]) if j.get("timings") else None
        out["queue_pos"] = None
        if j["status"] == "queued":
            waiting = [x["id"] for x in self.store.jobs_with_status("queued")]
            out["queue_pos"] = waiting.index(j["id"]) + 1 if j["id"] in waiting else None
        out["review_url"] = f"{self.base_url}/#job={j['id']}"
        out["tape"] = self._tape(j["id"])
        return out

    def _tape(self, jid: str) -> list[list]:
        """목록의 작은 테이프: 같은 화자가 이어 말한 구간 [시작, 끝, 화자, 모두 확인했나] (최대 400개)."""
        out: list[list] = []
        for s in self.store.segments(jid):
            last = out[-1] if out else None
            if last and last[2] == s["speaker"] and s["start"] - last[1] < 1.5:
                last[1] = max(last[1], s["end"])
                last[3] = last[3] and bool(s["checked"])
            else:
                out.append([round(s["start"], 1), round(s["end"], 1), s["speaker"], bool(s["checked"])])
        return out[:400]

    def jobs(self) -> list[dict]:
        return [self._summary(j) for j in self.store.jobs()]

    def job_detail(self, jid: str) -> dict:
        out = self.job_summary(jid)
        out["segments"] = self.store.segments(jid)
        out["speakers"] = self.store.speaker_names(jid)
        out["redoing"] = sorted(i for j, i in self.redoing if j == jid)
        return out

    def audio_path(self, jid: str) -> Path:
        j = self._job(jid)
        p = Path(j["audio_path"] or "")
        if not p.exists():
            raise ServiceError("소리 파일이 없습니다", 404)
        return p

    def export(self, jid: str, fmt: str) -> tuple[str, str, str]:
        d = self.job_detail(jid)
        segs, names = d["segments"], d["speakers"]
        stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in d["title"]).strip()[:80] or jid
        if fmt == "txt":
            return formats.to_txt(segs, names), "text/plain; charset=utf-8", f"{stem}.txt"
        if fmt == "srt":
            return formats.to_srt(segs, names), "application/x-subrip; charset=utf-8", f"{stem}.srt"
        if fmt == "json":
            meta = {k: d[k] for k in ("id", "title", "duration", "language", "engine", "model", "diarizer", "created")}
            return formats.to_json(meta, segs, names), "application/json", f"{stem}.json"
        raise ServiceError("형식은 txt·srt·json 중 하나")

    def search(self, q: str) -> list[dict]:
        q = q.strip()
        return self.store.search(q) if len(q) >= 2 else []

    # ---------------- 실시간 ----------------
    def start_live(self, *, source: str = "mic", url: Optional[str] = None, lang: Optional[str] = None, context: str = "",
                   title: Optional[str] = None, origin: str = "screen") -> dict:
        if source not in ("mic", "stream"):
            raise ServiceError("source 는 mic 또는 stream")
        if source == "stream" and not (url or "").strip():
            raise ServiceError("스트림 주소를 넣어 주세요")
        lang = (lang or "").strip().lower() or None
        sel = self.engine.select(lang, speakers=False)
        if not sel.asr:
            raise ServiceError(ENABLE_HINT, 503, "no_engine")
        jid = uuid.uuid4().hex[:12]
        wav = self.audio_dir / f"{jid}.wav"
        self.store.add_job(id=jid, title=(title or time.strftime("실시간 %m월 %d일 %H:%M"))[:200], kind="live", origin=origin,
                           status="live", stage=STAGES["load"], audio_path=str(wav), audio_name=wav.name, lang=lang,
                           context=(context or "")[:2000], speakers_on=1, source_url=url if source == "stream" else None,
                           engine=sel.asr[0], model=self.engine.model_of(sel.asr[0]), started=now())
        run = LiveRun(self, jid, sel.asr[0], source, url, wav, lang, context or "")
        self.lives[jid] = run
        run.start()
        self.notify(jid)
        return self.job_summary(jid)

    def feed_live(self, jid: str, pcm: bytes) -> None:
        run = self.lives.get(jid)
        if not run:
            raise ServiceError("듣는 중인 작업이 아닙니다", 409)
        run.feed(pcm)

    def stop_live(self, jid: str, final: bool = True) -> dict:
        run = self.lives.get(jid)
        if not run:
            raise ServiceError("듣는 중인 작업이 아닙니다", 409)
        run.stop(final)
        return self.job_summary(jid)

    def _live_ended(self, jid: str, recorded_s: float, final: bool, error: Optional[str]) -> None:
        self.lives.pop(jid, None)
        if error:
            self.store.update_job(jid, status="failed", stage=None, error=error, error_kind="failed", duration=recorded_s, finished=now())
        elif recorded_s <= 0:
            self.store.update_job(jid, status="failed", stage=None, error="들어온 소리가 없습니다", error_kind="bad_audio", finished=now())
        elif final:
            self.store.update_job(jid, status="queued", stage=STAGES["final"], duration=recorded_s)
            self.q.put(("job", jid))
        else:
            self.store.update_job(jid, status="done", stage=None, duration=recorded_s, finished=now(),
                                  warning="최종본 없이 멈춤 — 실시간 줄만 있습니다 (전체 다시 받아 적기로 최종본)")
        self.notify(jid)


class LiveRun:
    """실시간 하나: 엔진 프로세스 + 녹음 파일 + (stream 이면) ffmpeg. 엔진이 준비되기 전에 온 소리는 모아 뒀다 넘긴다."""

    def __init__(self, svc: Service, jid: str, engine_name: str, source: str, url: Optional[str], wav: Path,
                 lang: Optional[str], context: str):
        self.svc, self.jid, self.name, self.source, self.url, self.wav = svc, jid, engine_name, source, url, wav
        self.lang, self.context = lang, context
        self.lv = None
        self.pending: list[bytes] = []
        self.lock = threading.Lock()
        self.rec = wave.open(str(wav), "wb")
        self.rec.setnchannels(1), self.rec.setsampwidth(2), self.rec.setframerate(16000)
        self.bytes = 0
        self.stopping = threading.Event()
        self.final = True
        self.ff = None

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name=f"stt-live-{self.jid}").start()

    def feed(self, pcm: bytes) -> None:
        if self.stopping.is_set():
            return
        pcm = pcm[:len(pcm) // 2 * 2]
        with self.lock:
            self.rec.writeframes(pcm)
            self.bytes += len(pcm)
            if self.lv is None:
                self.pending.append(pcm)
                return
        self.lv.feed(pcm)

    def stop(self, final: bool = True) -> None:
        if self.stopping.is_set():   # 먼저 온 멈춤이 이긴다 — "최종본 없이" 뒤에 연결이 끊겨도 최종본으로 바꾸지 않음
            return
        self.final = final
        self.stopping.set()
        if self.ff:
            self.ff.terminate()
        with self.lock:
            if self.lv is not None:
                self.lv.close()

    def _run(self) -> None:
        import subprocess
        error = None
        try:
            lv = self.svc.engine.live(self.name, lang=self.lang, context=self.context)
            with self.lock:
                self.lv = lv
                for b in self.pending:
                    lv.feed(b)
                self.pending = []
                if self.stopping.is_set():
                    lv.close()
            self.svc.store.update_job(self.jid, stage=STAGES["listen"])
            self.svc.notify(self.jid)
            if self.source == "stream":
                self.ff = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", self.url, "-vn", "-ac", "1", "-ar", "16000",
                                            "-f", "s16le", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)

                def pump():
                    while not self.stopping.is_set():
                        b = self.ff.stdout.read(3200)
                        if not b:
                            break
                        self.feed(b)
                    self.stopping.set()
                    with self.lock:
                        lv.close()
                threading.Thread(target=pump, daemon=True).start()
            for d in lv.lines():
                self.svc.store.append_segment(self.jid, {"start": d["start"], "end": d["end"], "speaker": None,
                                                         "text": d["text"], "timing": "live"})
                self.svc.store.update_job(self.jid, duration=round(self.bytes / 32000, 1))
                self.svc.notify(self.jid)
            lv.wait()
        except EngineError as ex:
            error = ex.message
        except Exception as ex:  # noqa: BLE001
            error = f"실시간 내부 오류: {type(ex).__name__}: {ex}"[:300]
        finally:
            with self.lock:
                self.rec.close()
            if self.ff:
                self.ff.terminate()
            self.svc._live_ended(self.jid, round(self.bytes / 32000, 1), self.final, error)
