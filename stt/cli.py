"""CLI — 서버 없이 파일·실시간 입력을 받아 적어 파일로 남긴다 (스킬 `stt`가 부른다). 엔진은 engine.py 그대로.

    python -m stt transcribe <파일|주소> [...] [--lang ko] [--context "제목·용어"] [--num-speakers N] [--no-speakers]
                             [--engine <이름>] [--out-dir D] [--force] [--dry-run] [--allow-download]
    python -m stt live <ffmpeg 입력 | -> [--duration 초] [--realtime] [--no-final] [같은 선택지]

결과: <out-dir 또는 파일 옆>/<이름>.stt/ 에 segments.json · transcript.txt · subtitle.srt. 입력마다 한 줄 JSON 요약을 stdout 에.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

from stt.engine import Engine, EngineError, source_key
from stt.formats import to_json, to_srt, to_txt


def write_outputs(out: Path, meta: dict, segs: list[dict]) -> None:
    (out / "segments.json").write_text(to_json(meta, segs), encoding="utf-8")
    (out / "transcript.txt").write_text(to_txt(segs), encoding="utf-8")
    (out / "subtitle.srt").write_text(to_srt(segs), encoding="utf-8")


def process(src: Path, eng: Engine, a) -> dict:
    if not src.is_file():
        return {"source": str(src), "ok": False, "error": "파일 없음"}
    out = (Path(a.out_dir) if a.out_dir else src.parent) / f"{src.stem}.stt"
    sig = {"source": str(src.resolve()), "size": src.stat().st_size, "mtime": int(src.stat().st_mtime),
           "speakers": not a.no_speakers, "num_speakers": a.num_speakers, "lang": a.lang,
           "context": a.context or "", "engine_forced": a.engine}
    done = out / "segments.json"
    if done.exists() and not a.force:
        old = json.loads(done.read_text(encoding="utf-8"))["meta"]
        if all(old.get(k) == v for k, v in sig.items()):
            return {"source": str(src), "ok": True, "skipped": "이미 같은 조건으로 받아 적음 (--force 로 다시)",
                    "out": str(out), "engine": old["engine"]}
    sel = eng.select(a.lang, a.engine, speakers=not a.no_speakers)
    if a.dry_run:
        return {"source": str(src), "dry_run": True, "would_use": sel.asr[:1], "diarizer": sel.diarizer, "selection": sel.trail}
    if not sel.asr:
        return {"source": str(src), "ok": False, "error": "쓸 수 있는 받아 적기 엔진 없음", "selection": sel.trail}
    if not a.no_speakers and sel.diarizer is None:
        return {"source": str(src), "ok": False, "selection": sel.trail,
                "error": "화자 분리 엔진을 쓸 수 없음 — 고치거나 --no-speakers 로 (시각은 15초 조각 단위가 된다)"}
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="work-", dir=out))
    try:
        wav, _ = eng.prepare(src, work)
        pieces, n_turns, warning, t0 = None, None, None, time.perf_counter()
        if sel.diarizer:
            try:
                pieces = eng.diarize(wav, work, a.num_speakers)
                n_turns = len(pieces)
            except EngineError as ex:  # 말소리가 없으면 DiariZen 이 죽는다 (NOTES F5) — 화자 없이 계속하고 표시
                warning = f"화자 분리 실패 — 화자 없이 15초 조각으로 받아 적음: {ex.message[-160:]}"
        diar_s = round(time.perf_counter() - t0, 1)
        res, used = None, None
        for name in sel.asr:  # 고른 엔진이 실행 중 실패하면(메모리 부족 등) 다음 후보로 — 후보 수가 상한
            try:
                res, used = eng.transcribe(name, wav, pieces, lang=a.lang, context=a.context or ""), name
                break
            except EngineError as ex:
                next(t for t in sel.trail if t["engine"] == name)["run_error"] = ex.message[-300:]
        if res is None:
            return {"source": str(src), "ok": False, "error": "모든 후보 엔진 실행 실패", "selection": sel.trail}
        segs = res["segments"]
        meta = {**sig, "engine": used, "model": eng.model_of(used), "diarizer": sel.diarizer if not warning else None,
                "selection": sel.trail, "language": res.get("language"), "speaker_turns": n_turns,
                "pieces": res.get("pieces"), "audio_s": res.get("audio_s"), "diarize_s": diar_s,
                "load_s": res.get("load_s"), "run_s": res.get("run_s"), "warning": warning,
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        write_outputs(out, meta, segs)
        return {"source": str(src), "ok": True, **({"warning": warning} if warning else {}), "out": str(out),
                "engine": used, "diarizer": meta["diarizer"],
                "speakers": len({s.get("speaker") for s in segs if s.get("speaker") is not None}),
                "segments": len(segs), "audio_s": res.get("audio_s"), "diarize_s": diar_s, "run_s": res.get("run_s")}
    except EngineError as ex:
        return {"source": str(src), "ok": False, "error": ex.message, "selection": sel.trail}
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if out.exists() and not any(out.iterdir()):
            out.rmdir()  # 실패한 입력의 빈 폴더를 남기지 않는다


def live(eng: Engine, a) -> dict:
    """말이 멈출 때마다 받아 적어 한 줄씩 내고(live.jsonl), 들어온 소리는 live.wav 로 녹음.
    끝나면(소스 끝·--duration·Ctrl+C) 녹음본을 파일 방식으로 다시 받아 적어 최종본(화자 포함)."""
    out = Path(a.out_dir or f"live-{time.strftime('%Y%m%d-%H%M%S')}")
    if (out / "live.jsonl").exists() and not a.force:
        return {"live": True, "ok": False, "error": f"이미 있음: {out} (--force 로 덮어씀)"}
    sel = eng.select(a.lang, a.engine, speakers=False)
    if a.dry_run or not sel.asr:
        return {"live": True, "dry_run": a.dry_run, "would_use": sel.asr[:1], "selection": sel.trail,
                **({} if sel.asr or a.dry_run else {"ok": False, "error": "쓸 수 있는 받아 적기 엔진 없음"})}
    out.mkdir(parents=True, exist_ok=True)
    try:
        lv = eng.live(sel.asr[0], lang=a.lang, context=a.context or "")  # 모델을 다 올린 뒤에 소스를 연다
    except EngineError as ex:
        return {"live": True, "ok": False, "error": ex.message, "selection": sel.trail}
    ff = None
    if a.source == "-":
        src = sys.stdin.buffer
    else:  # ffmpeg 가 여는 무엇이든: 라이브 방송 주소, 파일(--realtime 이면 실제 속도), 장치
        ff = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error"] + (["-re"] if a.realtime else []) +
                              ["-i", a.source, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        src = ff.stdout
    stop = threading.Event()
    rec = wave.open(str(out / "live.wav"), "wb")
    rec.setnchannels(1), rec.setsampwidth(2), rec.setframerate(16000)
    limit = int(a.duration * 32000) if a.duration else None
    got = [0]

    def feed():
        try:
            while not stop.is_set():
                data = src.read(3200)
                if not data:
                    break
                if limit is not None:
                    data = data[:max(0, limit - got[0])]
                rec.writeframes(data)
                lv.feed(data)
                got[0] += len(data)
                if limit is not None and got[0] >= limit:
                    break
        except (BrokenPipeError, ValueError):
            pass
        finally:
            try:
                lv.close()
            except BrokenPipeError:
                pass

    th = threading.Thread(target=feed, daemon=True)
    th.start()
    n = 0
    with open(out / "live.jsonl", "w", encoding="utf-8") as jl:
        def take(d):
            line = json.dumps(d, ensure_ascii=False)
            jl.write(line + "\n")
            jl.flush()
            return line
        try:
            for d in lv.lines():
                sys.stdout.write(take(d) + "\n")
                sys.stdout.flush()
                n += 1
        except KeyboardInterrupt:  # 멈춤 → 소스를 닫고 엔진이 남은 말을 마저 내게 한다
            stop.set()
            if ff:
                ff.terminate()
            for d in lv.lines():
                take(d)
                n += 1
    th.join(timeout=5)
    rec.close()
    rc = lv.wait()
    ff_err = ""
    if ff:
        ff.terminate()
        ff_err = ff.stderr.read().decode("utf-8", "replace").strip()[-300:]
    rec_s = round(got[0] / 32000, 1)
    res = {"live": True, "ok": rc == 0 and rec_s > 0, "out": str(out), "engine": sel.asr[0], "utterances": n,
           "recorded_s": rec_s}
    if rec_s == 0:
        res["error"] = f"들어온 소리 없음{(' — ffmpeg: ' + ff_err) if ff_err else ''}"
    elif rc != 0:
        res["error"] = f"엔진 오류 — {lv.log.name}"
    if res["ok"] and not a.no_final:
        a.out_dir, a.force = str(out), True
        res["final"] = process(out / "live.wav", eng, a)
    return res


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lang", help="ISO 코드 (ko, en, ja …). 비우면 자동 감지")
    p.add_argument("--context", help="영상 제목·설명·용어 — 고유명사 받아 적기를 돕는다")
    p.add_argument("--num-speakers", type=int, help="알면 화자 수 (DiariZen 이 가장 가까운 수로 맞춘다)")
    p.add_argument("--no-speakers", action="store_true", help="화자 분리 생략 — 빠르지만 시각이 15초 조각 단위")
    p.add_argument("--engine", help="등록부의 받아 적기 엔진 이름으로 고정 (조건 검사는 그대로)")
    p.add_argument("--out-dir")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="어느 엔진을 왜 고르는지만 보여 줌")
    p.add_argument("--allow-download", action="store_true", help="로컬에 없는 모델을 받아도 됨 (기본: 오프라인)")


def fetch_url(url: str, eng: Engine, a) -> tuple[Path | None, dict]:
    """주소 → 소리 파일 (<out-dir 또는 현재 폴더>/<영상 id>.<확장자>, 옆에 <id>.info.json). 맥락이 비었으면 제목·채널·설명으로.
    같은 유튜브 영상(출처 키 — 서비스와 같은 규칙)을 이미 받았으면 다시 받지 않는다 (--force 면 다시)."""
    base = Path(a.out_dir or ".")
    key = source_key(url)
    vid = key.split(":", 1)[1] if key.startswith("youtube:") else None
    side = base / f"{vid}.info.json" if vid else None
    got_before = [p for p in base.glob(f"{vid}.*") if p.suffix != ".json" and p.is_file()] if vid else []
    if side and side.is_file() and got_before and not a.force:
        dst, info = got_before[0], json.loads(side.read_text(encoding="utf-8"))
    else:
        with tempfile.TemporaryDirectory(prefix="stt-fetch-") as td:
            try:
                got, info = eng.fetch(url, Path(td))
            except EngineError as ex:
                return None, {"source": url, "ok": False, "error": ex.message}
            base.mkdir(parents=True, exist_ok=True)
            dst = base / f"{info.get('id') or 'download'}{got.suffix}"
            shutil.move(str(got), str(dst))
        if info.get("id"):
            (base / f"{info['id']}.info.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    if not a.context:
        a.context = "\n".join(str(x) for x in (info.get("title"), info.get("channel") or info.get("uploader"),
                                                (info.get("description") or "")[:800]) if x)
    return dst, info


def run_transcribe(a) -> int:
    eng = Engine(allow_download=a.allow_download)
    if a.engine and a.engine not in eng.cfg["engines"]:
        print(json.dumps({"ok": False, "error": f"등록부에 없는 엔진: {a.engine}"}, ensure_ascii=False))
        return 2
    bad = 0
    ctx0 = a.context
    for raw in a.inputs:
        a.context = ctx0
        if raw.startswith(("http://", "https://")):
            if a.dry_run:
                sel = eng.select(a.lang, a.engine, speakers=not a.no_speakers)
                r = {"source": raw, "dry_run": True, "would_fetch": True, "would_use": sel.asr[:1], "diarizer": sel.diarizer,
                     "selection": sel.trail}
            else:
                src, info = fetch_url(raw, eng, a)
                r = info if src is None else {**process(src, eng, a), "url": raw, "title": info.get("title"),
                                              "video": {k: info.get(k) for k in ("id", "channel", "uploader", "upload_date",
                                                                                 "duration", "extractor", "webpage_url")}}
        else:
            r = process(Path(raw), eng, a)
        bad += r.get("ok") is False
        print(json.dumps(r, ensure_ascii=False), flush=True)
    return 1 if bad else 0


def run_live(a) -> int:
    r = live(Engine(allow_download=a.allow_download), a)
    print(json.dumps(r, ensure_ascii=False), flush=True)
    return 0 if r.get("ok") is not False and (r.get("final") or {}).get("ok", True) else 1
