"""엔진 (import 층) — 받아 적기를 단계별로 부른다. 표준 라이브러리만 쓰고, 모델은 등록부(models.json)의 환경으로 따로 띄운다.

    from stt.engine import Engine
    eng = Engine()
    sel = eng.select(lang="en")                    # 어느 엔진을 왜 고르나 (조건: 언어·모델 있음·환경·GPU 여유)
    wav, dur = eng.prepare(Path("a.m4a"), work)    # 16kHz mono wav
    turns = eng.diarize(wav, work)                 # 화자 구간 (말소리 없으면 EngineError kind="no_speech")
    res = eng.transcribe(sel.asr[0], wav, turns)   # 구간대로 받아 적기 → 문장 + 시각 + 화자
    with eng.live(sel.asr[0]) as lv: lv.feed(pcm); for u in lv.lines(): ...   # 실시간

단계마다 따로 부를 수 있어서 서비스가 "이 구간만 다시", "화자 분리 없이" 같은 일을 엔진을 다시 짜지 않고 한다.
긴 단계는 ``cancel`` (threading.Event) 을 받아, 켜지면 띄운 프로세스를 죽이고 EngineError(kind="canceled").
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Optional
from urllib.parse import parse_qs, urlparse

PKG = Path(__file__).resolve().parent
ENGINE_RUN = PKG / "engine_run.py"
MERGE_GAP_S = 1.0  # 같은 화자가 이보다 짧게 쉬고 이어 말하면 한 조각으로

Kind = Literal["no_engine", "no_speech", "bad_audio", "fetch", "failed", "canceled"]

_YT_HOSTS = ("youtube.com", "youtube-nocookie.com")
_YT_ID = re.compile(r"[A-Za-z0-9_-]+")


def source_key(url: str) -> str:
    """같은 영상 찾기용 출처 키. 유튜브는 영상 ID 로 (watch?v= · youtu.be/ · shorts/ · embed/ · live/, m.·www.,
    뒤의 &t= · &list= 같은 인자 무시) → "youtube:<ID>". 그 밖의 주소는 주소 그대로."""
    url = url.strip()
    u = urlparse(url)
    host = (u.hostname or "").lower()
    vid = None
    if host == "youtu.be":
        vid = u.path.strip("/").split("/")[0]
    elif any(host == h or host.endswith("." + h) for h in _YT_HOSTS):
        parts = [x for x in u.path.split("/") if x]
        if parts[:1] == ["watch"]:
            vid = (parse_qs(u.query).get("v") or [""])[0]
        elif len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
            vid = parts[1]
    if vid and _YT_ID.fullmatch(vid):
        return f"youtube:{vid}"
    return url


class EngineError(Exception):
    def __init__(self, kind: Kind, message: str):
        super().__init__(message)
        self.kind, self.message = kind, message


@dataclass
class Selection:
    asr: list[str]                     # 쓸 수 있는 받아 적기 엔진 (정확도 순)
    diarizer: Optional[str]            # 쓸 수 있는 화자 분리 엔진 (없거나 안 쓰면 None)
    trail: list[dict] = field(default_factory=list)   # [{engine, role?, why_not}]


def model_present(model: str) -> bool:
    """경로면 존재, HF id 면 로컬 캐시에 가중치(10MB 넘는 파일)가 있나."""
    p = Path(model).expanduser()
    if p.is_absolute():
        return p.exists()
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")) / "hub").expanduser()
    snaps = hub / f"models--{model.replace('/', '--')}" / "snapshots"
    return any(f.stat().st_size > 10_000_000 for f in snaps.glob("*/*") if f.is_file()) if snaps.exists() else False


def gpu_free_mib() -> Optional[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


class Engine:
    def __init__(self, config: Optional[Path] = None, *, allow_download: bool = False):
        self.config_path = Path(config or os.environ.get("STT_MODELS") or PKG / "models.json")
        self.cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.allow_download = allow_download
        self.env = {**os.environ, "PYTHONUTF8": "1"}
        if not allow_download:
            self.env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    # ---------------- 고르기 ----------------
    def why_not(self, name: str, lang: Optional[str], free: Optional[int]) -> Optional[str]:
        """이 엔진을 못 쓰는 이유. 쓸 수 있으면 None."""
        e = self.cfg["engines"].get(name)
        if e is None:
            return "등록부에 없음"
        langs = e.get("languages", "all")
        if lang and langs != "all" and lang not in self.cfg.get("languages", {}).get(langs, []):
            return f"언어 미지원 ({lang})"
        if not self.allow_download and not model_present(e["model"]):
            return f"모델 없음 ({e['model']} — 받지 않음)"
        for key in ("python", "command"):
            if key in e and not Path(e[key]).exists():
                return f"환경 없음 ({e[key]})"
        need = e.get("vram_mib", 0)
        if free is None:
            return "GPU 정보 없음 (nvidia-smi 실패)"
        if free < need:
            return f"GPU 여유 {free} MiB < 필요 {need} MiB"
        return None

    def select(self, lang: Optional[str] = None, engine: Optional[str] = None, speakers: bool = True) -> Selection:
        free = gpu_free_mib()
        names = [engine] if engine else self.cfg["prefer"]
        trail = [{"engine": n, "why_not": self.why_not(n, lang, free)} for n in names]
        diar = None
        if speakers:
            d = self.cfg.get("diarizer", "")
            why = self.why_not(d, None, free)
            trail.append({"engine": d, "role": "diarizer", "why_not": why})
            diar = None if why else d
        asr = [t["engine"] for t in trail if t["why_not"] is None and t.get("role") != "diarizer"]
        return Selection(asr, diar, trail)

    def model_of(self, name: str) -> str:
        return self.cfg["engines"][name]["model"]

    # ---------------- 단계 ----------------
    def _run(self, cmd: list[str], cancel: Optional[threading.Event]) -> subprocess.CompletedProcess:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=self.env,
                             start_new_session=True)
        box: dict = {}
        t = threading.Thread(target=lambda: box.update(zip(("out", "err"), p.communicate())), daemon=True)
        t.start()
        while t.is_alive():
            t.join(0.3)
            if cancel is not None and cancel.is_set():
                p.kill()
                t.join(5)
                raise EngineError("canceled", "취소함")
        return subprocess.CompletedProcess(cmd, p.returncode, box.get("out", ""), box.get("err", ""))

    def fetch(self, url: str, dest_dir: Path, *, cancel: Optional[threading.Event] = None) -> tuple[Path, dict]:
        """주소(유튜브 등 yt-dlp 가 여는 것) → 소리만 받아 dest_dir 에 저장. 먼저 제목·설명·길이를 읽는다.
        → (소리 파일, {title, description, uploader, channel, upload_date(YYYYMMDD), duration, webpage_url, extractor, id})"""
        if not url.startswith(("http://", "https://")):
            raise EngineError("fetch", "http(s) 주소만 받습니다")
        cmd0 = (self.cfg.get("downloader") or {}).get("command") or shutil.which("yt-dlp") or "yt-dlp"
        if not (Path(cmd0).exists() or shutil.which(cmd0)):
            raise EngineError("no_engine", f"주소를 받을 yt-dlp 가 없습니다 ({cmd0}) — models.json 의 downloader")
        r = self._run([cmd0, "-J", "--no-playlist", "--no-warnings", url], cancel)
        if r.returncode != 0:
            raise EngineError("fetch", f"주소를 열 수 없습니다: {(r.stderr.strip().splitlines() or ['?'])[-1][:200]}")
        meta = json.loads(r.stdout)
        if meta.get("is_live"):
            raise EngineError("fetch", "생방송 중인 주소입니다 — 실시간 받아 적기(스트림 주소)로 넣으세요")
        info = {k: meta.get(k) for k in ("title", "description", "uploader", "channel", "upload_date", "duration", "webpage_url",
                                          "extractor", "id")}
        dest_dir.mkdir(parents=True, exist_ok=True)
        r = self._run([cmd0, "-q", "--no-playlist", "--no-warnings", "-f", "bestaudio/best", "-o",
                       str(dest_dir / "source.%(ext)s"), url], cancel)
        got = sorted(dest_dir.glob("source.*"))
        if r.returncode != 0 or not got:
            raise EngineError("fetch", f"소리를 받지 못했습니다: {(r.stderr.strip().splitlines() or ['?'])[-1][:200]}")
        return got[0], info

    def prepare(self, src: Path, work: Path, *, start: Optional[float] = None, end: Optional[float] = None,
                name: str = "audio16k.wav") -> tuple[Path, float]:
        """ffmpeg 가 읽는 소리·영상 → 16kHz mono wav. start/end 를 주면 그 구간만."""
        dst = work / name
        cut = (["-ss", f"{start:.3f}"] if start is not None else []) + (["-to", f"{end:.3f}"] if end is not None else [])
        r = subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", *cut, "-i", str(src), "-vn", "-ac", "1", "-ar",
                            "16000", "-c:a", "pcm_s16le", str(dst)], capture_output=True, text=True)
        if r.returncode != 0 or not dst.exists() or dst.stat().st_size <= 44:
            raise EngineError("bad_audio", f"소리를 꺼낼 수 없는 파일입니다 ({r.stderr.strip()[-200:] or '소리 트랙 없음'})")
        return dst, (dst.stat().st_size - 44) / 32000

    def diarize(self, wav: Path, work: Path, num_speakers: Optional[int] = None,
                cancel: Optional[threading.Event] = None) -> list[dict]:
        name = self.cfg.get("diarizer")
        e = self.cfg["engines"].get(name or "")
        if not e:
            raise EngineError("no_engine", "등록부에 화자 분리 엔진이 없습니다")
        r = self._run([e["command"], str(wav), "--out-dir", str(work)] +
                      (["--num-speakers", str(num_speakers)] if num_speakers else []), cancel)
        rttms = sorted(work.glob(f"{wav.stem}*.rttm"))  # 화자 수를 주면 <이름>.n2.rttm 처럼 이름이 바뀐다
        if r.returncode != 0 or not rttms:
            if "negative dimensions" in r.stderr:     # 말소리 구간 0개에서 DiariZen 이 죽는다 (NOTES F5)
                raise EngineError("no_speech", "말소리를 찾지 못했습니다")
            raise EngineError("failed", f"화자 분리 실패: {r.stderr.strip()[-300:]}")
        turns = []
        for line in rttms[0].read_text().splitlines():
            p = line.split()
            if len(p) >= 8 and p[0] == "SPEAKER":
                s = float(p[3])
                turns.append({"start": s, "end": s + float(p[4]), "speaker": p[7]})
        turns.sort(key=lambda t: t["start"])
        merged: list[dict] = []
        for t in turns:
            if merged and merged[-1]["speaker"] == t["speaker"] and t["start"] - merged[-1]["end"] < MERGE_GAP_S:
                merged[-1]["end"] = max(merged[-1]["end"], t["end"])
            else:
                merged.append(dict(t))
        return merged

    def transcribe(self, name: str, wav: Path, pieces: Optional[list[dict]] = None, *, lang: Optional[str] = None,
                   context: str = "", cancel: Optional[threading.Event] = None) -> dict:
        """pieces = 화자 구간 [{start,end,speaker}] (없으면 전체를 조용한 지점에서 15초 이하로).
        → {"language", "segments": [{start,end,speaker,text,timing}], "pieces", "load_s", "run_s", "audio_s"}"""
        e = self.cfg["engines"][name]
        with tempfile.TemporaryDirectory(prefix="stt-") as td:
            out = Path(td) / "out.json"
            cmd = [e["python"], str(ENGINE_RUN), "--runner", e["runner"], "--model", e["model"], "--audio", str(wav),
                   "--out", str(out), "--context", context or ""] + (["--lang", lang] if lang else [])
            if pieces is not None:
                pf = Path(td) / "pieces.json"
                pf.write_text(json.dumps(pieces), encoding="utf-8")
                cmd += ["--pieces", str(pf)]
            r = self._run(cmd, cancel)
            if r.returncode != 0 or not out.exists():
                raise EngineError("failed", f"받아 적기 실패 ({name}): {r.stderr.strip()[-300:]}")
            return json.loads(out.read_text(encoding="utf-8"))

    def live(self, name: str, *, lang: Optional[str] = None, context: str = "") -> "Live":
        e = self.cfg["engines"][name]
        return Live([e["python"], str(ENGINE_RUN), "--live", "--runner", e["runner"], "--model", e["model"],
                     "--context", context or ""] + (["--lang", lang] if lang else []), self.env)


class Live:
    """실시간 엔진 프로세스 하나. feed(PCM16 LE 16kHz mono) → lines() 가 발화마다 dict 를 낸다. close() 하면 남은 말을 마저 낸다."""

    def __init__(self, cmd: list[str], env: dict):
        self.log = tempfile.NamedTemporaryFile(prefix="stt-live-", suffix=".log", delete=False)
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=env,
                                  start_new_session=True)
        first = self.p.stdout.readline()           # 모델을 다 올리면 {"ready": true}
        if b'"ready"' not in first:
            self.p.kill()
            raise EngineError("failed", f"실시간 엔진 시작 실패 — {self.log.name}")
        self.load_s = json.loads(first).get("load_s")
        self._lock = threading.Lock()

    def feed(self, pcm: bytes) -> None:
        with self._lock:
            if self.p.stdin and not self.p.stdin.closed:
                self.p.stdin.write(pcm)
                self.p.stdin.flush()

    def lines(self) -> Iterator[dict]:
        for raw in self.p.stdout:
            try:
                d = json.loads(raw)
            except ValueError:
                continue
            if "text" in d:
                yield d

    def close(self) -> None:
        with self._lock:
            if self.p.stdin and not self.p.stdin.closed:
                self.p.stdin.close()

    def kill(self) -> None:
        self.p.kill()

    def wait(self) -> int:
        return self.p.wait()

    def __enter__(self) -> "Live":
        return self

    def __exit__(self, *a) -> None:
        self.close()
