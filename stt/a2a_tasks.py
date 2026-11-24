"""A2A 창구 — 다른 에이전트가 받아 적기를 맡긴다 (Agent Card + JSON-RPC, REF/AGENT_PRODUCT.md).

배관(디스커버리·JSON-RPC·에러 코드)은 doc_parser 의 도메인 없는 ``a2a.py`` 를 그대로 쓴다. 여기는 stt 일만:
메시지 → 서비스 함수. 화면과 같은 서비스라 A2A 로 맡긴 일도 작업 목록에 보이고, 사람이 이어서 확인·고친다.

기능 transcribe — 소리·영상 하나를 받아 적는다.
  보내는 것: 파일 part ({"raw": base64, "filename"}) 또는 {"url": "file:///절대경로" | "https://영상 주소"} 또는
  data {"path": "/절대경로"} / data {"url": "https://…"} (유튜브 등 yt-dlp 가 여는 주소 — 소리만 받는다).
  선택 (data): lang(ISO), context(제목·용어), numSpeakers, speakers(false면 화자 분리 생략), title.
  돌려받는 것: 처음엔 WORKING (줄에 들어감). GetTask 로 다시 물으면 끝났을 때 COMPLETED + artifact
  (text = 시각·화자가 붙은 글, data = {segments, speakers, meta}) + metadata.reviewUrl (사람이 확인·고치는 화면).
  같은 messageId 로 두 번 보내면 처음 작업을 돌려준다. CancelTask 는 줄에 있거나 받아 적는 중일 때만.
"""
from __future__ import annotations

import base64
import binascii
import tempfile
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

from doc_parser import a2a

from stt import formats
from stt.service import Service, ServiceError

SKILLS = ("transcribe",)
STATE = {"queued": "TASK_STATE_WORKING", "running": "TASK_STATE_WORKING", "live": "TASK_STATE_WORKING",
         "done": "TASK_STATE_COMPLETED", "failed": "TASK_STATE_FAILED", "canceled": "TASK_STATE_CANCELED",
         "rejected": "TASK_STATE_REJECTED"}

CARD = {
    "name": "stt",
    "description": "이 PC의 로컬 모델로 소리·영상을 받아 적는다 (화자 구분·문장 시각 포함). 결과는 사람이 확인·고치는 화면 링크와 함께.",
    "version": "1.0.0",
    "provider": {"organization": "side_projects", "url": "http://127.0.0.1:8773"},
    "defaultInputModes": ["audio/*", "video/*", "application/json", "text/plain"],
    "defaultOutputModes": ["text/plain", "application/json"],
    "skills": [{
        "id": "transcribe", "name": "받아 적기",
        "description": "소리·영상 하나를 받아 적는다. 파일 part(raw base64) / file:// url / data.path(이 PC의 절대 경로) / "
                       "https 주소(url part 또는 data.url — 유튜브 등, 소리만 받음, 제목·설명을 맥락으로). "
                       "data: lang, context, numSpeakers, speakers, title. 끝나면 artifact 에 글(text)과 구간(data), metadata.reviewUrl.",
        "tags": ["stt", "transcription", "speaker-diarization"],
        "examples": ['{"data": {"path": "/abs/podcast.m4a", "lang": "en", "context": "English Cactus podcast"}}'],
        "inputModes": ["audio/*", "video/*", "application/json"], "outputModes": ["text/plain", "application/json"],
    }],
}


def _parts(message: dict) -> tuple[str, dict, list[tuple[str, bytes | Path | str]]]:
    text, data, files = [], {}, []
    for p in message.get("parts") or []:
        if not isinstance(p, dict):
            continue
        if "text" in p:
            text.append(str(p["text"]))
        elif "data" in p and isinstance(p["data"], dict):
            data.update(p["data"])
        elif "raw" in p:
            try:
                files.append((str(p.get("filename") or "audio.bin"), base64.b64decode(p["raw"], validate=True)))
            except (binascii.Error, ValueError) as exc:
                raise a2a.invalid_params("raw is not valid base64") from exc
        elif "url" in p:
            u = urlparse(str(p["url"]))
            if u.scheme == "file":
                files.append((Path(unquote(u.path)).name, Path(unquote(u.path))))
            elif u.scheme in ("http", "https"):
                files.append(("", str(p["url"])))
            else:
                raise a2a.content_type_not_supported("url parts: file:// (이 PC) 또는 http(s) 주소")
    if data.get("path"):
        files.append((Path(str(data["path"])).name, Path(str(data["path"]))))
    if data.get("url"):
        files.append(("", str(data["url"])))
    return "\n".join(text).strip(), data, files


class SttTasks:
    """TaskHandler: 한 A2A 작업 = 서비스 작업 하나. 상태는 서비스의 작업 상태에서 읽는다 (따로 두지 않음)."""

    def __init__(self, svc: Service):
        self.svc, self.store = svc, svc.store

    def send(self, message: dict, configuration: dict, base: str) -> dict:
        if message.get("taskId"):
            raise a2a.invalid_params("이 작업은 더 받을 것이 없습니다 — 새 메시지로 보내세요")
        mid = message.get("messageId")
        if mid:
            old = self.store.task_by_message(str(mid))
            if old:                                      # 같은 요청 2회 → 처음 작업 (다시 받아 적지 않음)
                return self._view(old, base)
        text, data, files = _parts(message)
        skill = data.get("skill") or (message.get("metadata") or {}).get("skill") or "transcribe"
        tid = uuid.uuid4().hex[:16]
        ctx = str(message.get("contextId") or tid)
        try:
            if skill not in SKILLS:
                raise a2a.invalid_params(f"skill 은 {', '.join(SKILLS)}")
            if len(files) != 1:
                raise a2a.invalid_params("소리·영상 하나를 보내세요 (파일 part / file:// 또는 https 주소 / data.path / data.url)")
            name, body = files[0]
            caller = str((message.get("metadata") or {}).get("caller") or data.get("caller") or "") or None
            opts = dict(title=data.get("title"), lang=data.get("lang"), context=str(data.get("context") or text or ""),
                        num_speakers=int(data["numSpeakers"]) if data.get("numSpeakers") else None,
                        speakers=data.get("speakers", True) is not False, origin="a2a", caller=caller)
            if isinstance(body, str):
                job = self.svc.create_url_job(body, **{k: v for k, v in opts.items()})
            elif isinstance(body, Path):
                if not body.is_absolute() or not body.is_file():
                    raise a2a.invalid_params(f"파일이 없습니다: {body} (이 PC의 절대 경로)")
                job = self.svc.create_file_job(body, name, **opts)
            else:
                with tempfile.NamedTemporaryFile(suffix=Path(name).suffix, delete=False) as f:
                    f.write(body)
                job = self.svc.create_file_job(Path(f.name), name, move=True, **opts)
        except a2a.A2AError as exc:                      # 형식 오류 → 거부로 남기고 JSON-RPC 에러로도 알린다
            self.store.add_task(tid, ctx, str(mid) if mid else None, None, state="rejected", error=exc.message)
            raise
        except ServiceError as exc:
            self.store.add_task(tid, ctx, str(mid) if mid else None, None, state="rejected", error=exc.message)
            return self._view(self.store.task(tid), base)
        self.store.add_task(tid, ctx, str(mid) if mid else None, job["id"])
        return self._view(self.store.task(tid), base)

    def get(self, task_id: str, base: str) -> dict:
        return self._view(self._task(task_id), base)

    def list(self, params: dict, base: str) -> list[dict]:
        want = params.get("status")
        views = [self._view(t, base) for t in self.store.tasks(params.get("contextId"))]
        return [v for v in views if not want or v["status"]["state"] == want]

    def cancel(self, task_id: str, base: str) -> dict:
        t = self._task(task_id)
        view = self._view(t, base)
        if view["status"]["state"] != "TASK_STATE_WORKING":
            raise a2a.not_cancelable(task_id, view["status"]["state"])
        self.svc.cancel(t["job_id"])
        return self._view(t, base)

    def _task(self, tid: str) -> dict:
        t = self.store.task(tid)
        if not t:
            raise a2a.task_not_found(tid)
        return t

    def _view(self, t: dict, base: str) -> dict:
        if t["state"] == "rejected" or not t["job_id"]:
            return a2a.task(t["id"], t["context_id"], "TASK_STATE_REJECTED", text=t["error"] or "거부됨")
        j = self.store.job(t["job_id"])
        if not j:
            return a2a.task(t["id"], t["context_id"], "TASK_STATE_CANCELED", text="화면에서 지운 작업입니다")
        state = STATE[j["status"]]
        review = f"{base}/#job={j['id']}"
        meta = {"jobId": j["id"], "reviewUrl": review, "status": j["status"], "stage": j["stage"]}
        if state == "TASK_STATE_WORKING":
            msg = j["stage"] or ("줄에서 기다리는 중" if j["status"] == "queued" else "받아 적는 중")
            return a2a.task(t["id"], t["context_id"], state, text=msg, metadata=meta)
        if state == "TASK_STATE_FAILED":
            return a2a.task(t["id"], t["context_id"], state, text=j["error"] or "실패", metadata={**meta, "errorKind": j["error_kind"]})
        if state == "TASK_STATE_CANCELED":
            return a2a.task(t["id"], t["context_id"], state, text="취소됨", metadata=meta)
        d = self.svc.job_detail(j["id"])
        segs, names = d["segments"], d["speakers"]
        info = {k: d[k] for k in ("title", "duration", "language", "engine", "model", "diarizer", "warning", "n_checked", "n_edited")}
        artifact = {"artifactId": f"{j['id']}-transcript", "name": "transcript",
                    "parts": [{"text": formats.to_txt(segs, names) if segs else ""},
                              {"data": {"segments": [{k: s[k] for k in ("start", "end", "speaker", "text", "timing")} for s in segs],
                                        "speakers": names, "meta": info}}]}
        summary = (f"{len(segs)}문장 받아 적음" if segs else "말소리를 찾지 못했습니다") + f" — 확인·고치기: {review}"
        return a2a.task(t["id"], t["context_id"], state, text=summary, artifacts=[artifact], metadata=meta)


def mount(app, svc: Service) -> None:
    a2a.mount_a2a(app, CARD, SttTasks(svc))
