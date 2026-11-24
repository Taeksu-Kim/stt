"""받아 적은 글의 내보내기 형식 — CLI 결과 파일과 화면·A2A 내려받기가 같은 함수를 쓴다."""
from __future__ import annotations

import json
from typing import Optional


def hms(t: float, sep: str = "") -> str:
    """초 → 00:01:23 (sep 를 주면 밀리초까지, 자막용 00:01:23,450)."""
    ms = int(round(max(0.0, t) * 1000))
    base = f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02}"
    return f"{base}{sep}{ms % 1000:03}" if sep else base


def speaker_name(label: Optional[str], names: Optional[dict] = None) -> str:
    if label is None:
        return ""
    if names and names.get(label):
        return names[label]
    return f"화자 {label}" if str(label).isdigit() else str(label)


def to_txt(segs: list[dict], names: Optional[dict] = None) -> str:
    lines, prev = [], object()
    for s in segs:
        if s.get("speaker") is not None and s["speaker"] != prev:
            lines.append(f"\n[{speaker_name(s['speaker'], names)}]")
            prev = s["speaker"]
        lines.append(f"[{hms(s['start'])}] {s['text']}")
    return "\n".join(lines).strip() + "\n"


def to_srt(segs: list[dict], names: Optional[dict] = None) -> str:
    blocks = []
    for i, s in enumerate(segs, 1):
        who = speaker_name(s.get("speaker"), names)
        blocks.append(f"{i}\n{hms(s['start'], ',')} --> {hms(s['end'], ',')}\n{'[' + who + '] ' if who else ''}{s['text']}\n")
    return "\n".join(blocks)


def to_json(meta: dict, segs: list[dict], names: Optional[dict] = None) -> str:
    return json.dumps({"meta": meta, "speakers": names or {}, "segments": segs}, ensure_ascii=False, indent=1)
