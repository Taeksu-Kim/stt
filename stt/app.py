"""제품 API + 화면 + A2A — 모두 같은 서비스(service.py)를 부른다 (REF/AGENT_PRODUCT.md)."""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from stt.service import EngineLike, Service, ServiceError

STATIC = Path(__file__).resolve().parent / "static"


class SegmentPatch(BaseModel):
    text: Optional[str] = Field(default=None, max_length=4000)
    checked: Optional[bool] = None


class TitleBody(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class SpeakerBody(BaseModel):
    name: str = Field(default="", max_length=60)


class UrlBody(BaseModel):
    url: str = Field(min_length=8, max_length=2000)
    title: Optional[str] = Field(default=None, max_length=200)
    lang: Optional[str] = Field(default=None, max_length=8)
    context: str = Field(default="", max_length=2000)
    num_speakers: Optional[int] = Field(default=None, ge=1, le=20)
    speakers: bool = True
    force: bool = False          # true 면 같은 영상을 이미 받아 적었어도 새 작업으로


class MergeBody(BaseModel):
    into: str = Field(min_length=1, max_length=40)


class LiveBody(BaseModel):
    source: Literal["mic", "stream"] = "mic"
    url: Optional[str] = Field(default=None, max_length=2000)
    lang: Optional[str] = Field(default=None, max_length=8)
    context: str = Field(default="", max_length=2000)
    title: Optional[str] = Field(default=None, max_length=200)


def create_app(data_dir: str | Path, *, engine: Optional[EngineLike] = None, service: Optional[Service] = None,
               self_url: str = "http://127.0.0.1:8773") -> FastAPI:
    """``engine``: 받아 적기 엔진 주입 (테스트는 가짜). 없으면 등록부(models.json)대로 — 모델·환경이 없어도 서버는 뜨고
    받아 적기만 "켜는 법"을 안내한다."""
    if service is None:
        if engine is None:
            from stt.engine import Engine
            engine = Engine()
        service = Service(Path(data_dir), engine, base_url=self_url)
    svc = service
    app = FastAPI(title="stt", docs_url=None, redoc_url=None)
    app.state.service = svc

    @app.exception_handler(ServiceError)
    def service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse({"detail": {"kind": exc.kind, "message": exc.message}}, status_code=exc.http)

    from stt.a2a_tasks import mount   # A2A 창구 — 같은 서비스
    mount(app, svc)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    def status(fresh: bool = False) -> dict:
        return svc.status(fresh)

    @app.get("/api/jobs")
    def jobs() -> list[dict]:
        return svc.jobs()

    @app.post("/api/jobs")
    async def upload(files: list[UploadFile] = File(...), lang: str = Form(""), context: str = Form(""),
                     num_speakers: str = Form(""), speakers: str = Form("1")) -> list[dict]:
        out = []
        for f in files:
            with tempfile.NamedTemporaryFile(suffix=Path(f.filename or "").suffix, delete=False) as tmp:
                while chunk := await f.read(1 << 20):
                    tmp.write(chunk)
            n = int(num_speakers) if num_speakers.strip().isdigit() and int(num_speakers) > 0 else None
            out.append(svc.create_file_job(Path(tmp.name), f.filename or "audio", lang=lang, context=context,
                                           num_speakers=n, speakers=speakers != "0", move=True))
        return out

    @app.post("/api/jobs/url")
    def upload_url(body: UrlBody) -> dict:
        return svc.create_url_job(body.url, title=body.title, lang=body.lang, context=body.context,
                                  num_speakers=body.num_speakers, speakers=body.speakers, force=body.force)

    @app.get("/api/jobs/{jid}")
    def job(jid: str) -> dict:
        return svc.job_detail(jid)

    @app.patch("/api/jobs/{jid}")
    def rename(jid: str, body: TitleBody) -> dict:
        return svc.rename(jid, body.title)

    @app.delete("/api/jobs/{jid}")
    def delete(jid: str) -> dict:
        svc.delete(jid)
        return {"deleted": jid}

    @app.post("/api/jobs/{jid}/cancel")
    def cancel(jid: str) -> dict:
        return svc.cancel(jid)

    @app.post("/api/jobs/{jid}/retranscribe")
    def retranscribe(jid: str) -> dict:
        return svc.retranscribe(jid)

    @app.patch("/api/jobs/{jid}/segments/{idx}")
    def patch_segment(jid: str, idx: int, body: SegmentPatch) -> dict:
        return svc.edit_segment(jid, idx, text=body.text, checked=body.checked)

    @app.post("/api/jobs/{jid}/segments/{idx}/revert")
    def revert_segment(jid: str, idx: int) -> dict:
        return svc.revert_segment(jid, idx)

    @app.post("/api/jobs/{jid}/segments/{idx}/redo")
    def redo_segment(jid: str, idx: int) -> dict:
        return svc.redo_segment(jid, idx)

    @app.post("/api/jobs/{jid}/check-until/{idx}")
    def check_until(jid: str, idx: int) -> dict:
        return svc.check_until(jid, idx)

    @app.put("/api/jobs/{jid}/speakers/{label}")
    def speaker(jid: str, label: str, body: SpeakerBody) -> dict:
        return svc.rename_speaker(jid, label, body.name)

    @app.post("/api/jobs/{jid}/speakers/{label}/merge")
    def merge_speaker(jid: str, label: str, body: MergeBody) -> dict:
        return svc.merge_speaker(jid, label, body.into)

    @app.get("/api/jobs/{jid}/audio")
    def audio(jid: str) -> FileResponse:
        return FileResponse(svc.audio_path(jid))

    @app.get("/api/jobs/{jid}/export")
    def export(jid: str, fmt: str = "txt") -> Response:
        body, media, name = svc.export(jid, fmt)
        from urllib.parse import quote
        return Response(body, media_type=media,
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})

    @app.get("/api/search")
    def search(q: str = "") -> list[dict]:
        return svc.search(q)

    @app.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        async def gen():
            v = svc.version
            yield f"data: {json.dumps({'v': v, 'jobs': []})}\n\n"
            while not await request.is_disconnected():
                nv, ids = await asyncio.to_thread(svc.wait_change, v, 15.0)
                if nv != v:
                    v = nv
                    yield f"data: {json.dumps({'v': v, 'jobs': ids})}\n\n"
                else:
                    yield ": ping\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})

    @app.post("/api/live")
    def start_live(body: LiveBody) -> dict:
        return svc.start_live(source=body.source, url=body.url, lang=body.lang, context=body.context, title=body.title)

    @app.post("/api/live/{jid}/stop")
    def stop_live(jid: str, final: bool = True) -> dict:
        return svc.stop_live(jid, final)

    @app.websocket("/api/live/{jid}/ws")
    async def live_ws(ws: WebSocket, jid: str) -> None:
        """브라우저 마이크 → PCM16 LE 16kHz mono 바이너리. 글자 'stop' 이면 끝내고 최종본. 끊기면 그때까지로 최종본."""
        await ws.accept()
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    svc.feed_live(jid, msg["bytes"])
                elif msg.get("text") == "stop":
                    break
        except (WebSocketDisconnect, ServiceError):
            pass
        finally:
            if jid in svc.lives:
                svc.stop_live(jid, True)
            try:
                await ws.close()
            except RuntimeError:
                pass

    return app
