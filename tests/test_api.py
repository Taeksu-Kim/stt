"""제품 API · A2A 진입점 — 모델만 가짜 (CLAUDE.md: 자동 테스트는 진입점 단위, 외부 모델만 가짜).

부정 케이스 먼저: 같은 요청 2회 / 빈·없는 파일 / 여러 파일 / 취소 / 재시작 후 이어서 / 엔진 꺼짐.
"""
from __future__ import annotations

import base64
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stt.app import create_app
from stt.engine import EngineError, Selection
from stt.service import Service


class FakeLive:
    def __init__(self):
        self.buf, self.closed, self.out = 0, threading.Event(), []

    def feed(self, pcm: bytes):
        self.buf += len(pcm)

    def lines(self):
        n = 0
        while not self.closed.is_set() or n * 32000 < self.buf:
            if (n + 1) * 32000 <= self.buf or (self.closed.is_set() and n * 32000 < self.buf):
                yield {"start": float(n), "end": float(n + 1), "text": f"실시간 {n}"}
                n += 1
            else:
                time.sleep(0.01)

    def close(self):
        self.closed.set()

    def wait(self):
        return 0


class FakeEngine:
    def __init__(self, on=True, diar=True):
        self.on, self.diar = on, diar
        self.block = threading.Event()        # set 이면 transcribe 가 풀릴 때까지 기다린다 (취소 시험)
        self.block_fetch = threading.Event()  # set 이면 fetch 가 풀릴 때까지 기다린다
        self.release = threading.Event()
        self.calls = []

    def select(self, lang=None, engine=None, speakers=True):
        asr = ["fake-asr"] if self.on else []
        d = "fake-diar" if (speakers and self.diar and self.on) else None
        return Selection(asr, d, [{"engine": "fake-asr", "why_not": None if self.on else "모델 없음"}])

    def model_of(self, name):
        return "fake/model"

    def prepare(self, src, work, *, start=None, end=None, name="audio16k.wav"):
        if Path(src).read_bytes()[:4] == b"BAD!":
            raise EngineError("bad_audio", "소리를 꺼낼 수 없는 파일입니다")
        p = work / name
        p.write_bytes(b"x")
        return p, 12.0 if start is None else (end - start)

    def diarize(self, wav, work, num_speakers=None, cancel=None):
        return [{"start": 0.0, "end": 5.0, "speaker": "1"}, {"start": 6.0, "end": 12.0, "speaker": "0"}]

    def transcribe(self, name, wav, pieces=None, *, lang=None, context="", cancel=None):
        self.calls.append(("transcribe", pieces is not None, context))
        if self.block.is_set():
            while not self.release.wait(0.05):
                if cancel is not None and cancel.is_set():
                    raise EngineError("canceled", "취소함")
        if pieces is None:
            return {"language": "ko", "segments": [{"start": 0.0, "end": 2.0, "speaker": None, "text": "다시 받아 적은 글", "timing": "piece"}]}
        segs = [{"start": p["start"], "end": p["end"], "speaker": p["speaker"], "text": f"{p['speaker']}번 화자의 말.",
                 "timing": "piece"} for p in pieces]
        return {"language": lang or "en", "segments": segs, "load_s": 0.1, "run_s": 0.2, "audio_s": 12.0}

    def live(self, name, *, lang=None, context=""):
        return FakeLive()

    def fetch(self, url, dest_dir, *, cancel=None):
        self.calls.append(("fetch", url))
        if "bad" in url:
            raise EngineError("fetch", "주소를 열 수 없습니다: 404")
        if self.block_fetch.is_set():
            while not self.release.wait(0.05):
                if cancel is not None and cancel.is_set():
                    raise EngineError("canceled", "취소함")
        p = dest_dir / "source.webm"
        p.write_bytes(b"audio")
        return p, {"title": "DATING | Intermediate english podcast", "uploader": "English Cactus Podcast",
                   "description": "learn English", "duration": 12.0, "id": "DHRm_2A2Qag"}


def wait_status(c: TestClient, jid: str, want: str, timeout=5.0) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] == want:
            return j
        time.sleep(0.03)
    raise AssertionError(f"{jid} status {j['status']} != {want}")


@pytest.fixture
def env(tmp_path):
    eng = FakeEngine()
    app = create_app(tmp_path / "data", engine=eng)
    with TestClient(app) as c:
        yield c, eng, tmp_path


def upload(c, name="a.m4a", body=b"RIFF....", **form):
    r = c.post("/api/jobs", files=[("files", (name, body, "audio/mp4"))], data=form)
    assert r.status_code == 200, r.text
    return r.json()[0]


def test_upload_to_done_and_human_actions(env):
    c, eng, _ = env
    job = upload(c, context="English Cactus")
    j = wait_status(c, job["id"], "done")
    assert [s["speaker"] for s in j["segments"]] == ["1", "0"]
    assert j["diarizer"] == "fake-diar" and eng.calls[-1] == ("transcribe", True, "English Cactus")
    # 고치기 → 확인으로, 원래 글은 asr_text 에 남음
    s = c.patch(f"/api/jobs/{job['id']}/segments/0", json={"text": "And my name is Brian."}).json()
    assert s["text"] == "And my name is Brian." and s["asr_text"] == "1번 화자의 말." and s["checked"] == 1
    assert c.patch(f"/api/jobs/{job['id']}/segments/0", json={"text": "   "}).status_code == 400
    assert c.post(f"/api/jobs/{job['id']}/segments/0/revert").json()["text"] == "1번 화자의 말."
    c.post(f"/api/jobs/{job['id']}/check-until/1")
    assert c.get("/api/jobs").json()[0]["n_checked"] == 2
    c.put(f"/api/jobs/{job['id']}/speakers/1", json={"name": "Brian"})
    txt = c.get(f"/api/jobs/{job['id']}/export?fmt=txt").text
    assert "[Brian]" in txt and "[화자 0]" in txt
    assert "00:00:00,000 --> 00:00:05,000" in c.get(f"/api/jobs/{job['id']}/export?fmt=srt").text
    assert c.get(f"/api/jobs/{job['id']}/export?fmt=doc").status_code == 400
    assert c.get("/api/search?q=화자의").json()[0]["job_id"] == job["id"]
    assert c.get("/api/search?q=x").json() == []           # 한 글자는 찾지 않음
    # 구간 다시 받아 적기
    c.post(f"/api/jobs/{job['id']}/segments/1/redo")
    end = time.time() + 5
    while c.get(f"/api/jobs/{job['id']}").json()["segments"][1]["text"] != "다시 받아 적은 글":
        assert time.time() < end
        time.sleep(0.03)
    assert c.patch(f"/api/jobs/{job['id']}", json={"title": "새 제목"}).json()["title"] == "새 제목"
    # 화자 합치기: 0 → 1 (이름은 1 쪽), 같은 화자·없는 화자는 거절
    assert c.post(f"/api/jobs/{job['id']}/speakers/0/merge", json={"into": "0"}).status_code == 400
    assert c.post(f"/api/jobs/{job['id']}/speakers/9/merge", json={"into": "1"}).status_code == 400
    assert c.post(f"/api/jobs/{job['id']}/speakers/0/merge", json={"into": "1"}).json()["moved"] == 1
    d = c.get(f"/api/jobs/{job['id']}").json()
    assert {s["speaker"] for s in d["segments"]} == {"1"} and d["speakers"] == {"1": "Brian"}


def test_empty_bad_and_multiple_files(env):
    c, _, _ = env
    assert c.post("/api/jobs", files=[("files", ("e.wav", b"", "audio/wav"))]).status_code == 400
    r = c.post("/api/jobs", files=[("files", ("a.wav", b"1234", "audio/wav")), ("files", ("b.wav", b"5678", "audio/wav"))])
    assert len(r.json()) == 2
    bad = upload(c, "bad.mp3", b"BAD!xxxx")
    j = wait_status(c, bad["id"], "failed")
    assert j["error_kind"] == "bad_audio"
    for x in r.json():
        wait_status(c, x["id"], "done")


def test_cancel_running_and_queued_then_retranscribe(env):
    c, eng, _ = env
    eng.block.set()
    a, b = upload(c, "a.wav", b"aaaa"), upload(c, "b.wav", b"bbbb")
    wait_status(c, a["id"], "running")
    assert c.get(f"/api/jobs/{b['id']}").json()["queue_pos"] == 1
    c.post(f"/api/jobs/{b['id']}/cancel")
    c.post(f"/api/jobs/{a['id']}/cancel")
    wait_status(c, a["id"], "canceled")
    assert c.get(f"/api/jobs/{b['id']}").json()["status"] == "canceled"
    assert c.post(f"/api/jobs/{a['id']}/cancel").status_code == 409
    eng.block.clear()
    c.post(f"/api/jobs/{a['id']}/retranscribe")
    wait_status(c, a["id"], "done")
    assert c.delete(f"/api/jobs/{a['id']}").status_code == 200
    assert c.get(f"/api/jobs/{a['id']}").status_code == 404


def test_restart_resumes_queue(tmp_path):
    eng = FakeEngine()
    svc = Service(tmp_path / "d", eng, start_worker=False)       # 일꾼 없이 줄에만 넣고 "꺼짐"
    j = svc.create_file_job(_file(tmp_path, b"abcd"), "x.wav")
    svc.store.update_job(j["id"], status="running")               # 돌던 중 꺼진 것처럼
    with TestClient(create_app(tmp_path / "d", engine=eng)) as c:
        wait_status(c, j["id"], "done")


def test_engine_off_server_up_and_failed_job(tmp_path):
    with TestClient(create_app(tmp_path / "d", engine=FakeEngine(on=False))) as c:
        st = c.get("/api/status?fresh=1").json()
        assert st["ok"] is False and "dry-run" in st["hint"]
        job = upload(c)
        j = wait_status(c, job["id"], "failed")
        assert j["error_kind"] == "no_engine"
        assert c.post("/api/live", json={"source": "mic"}).status_code == 503


def test_a2a_same_message_twice_cancel_and_review_url(env):
    c, eng, tmp = env
    card = c.get("/.well-known/agent-card.json").json()
    assert card["skills"][0]["id"] == "transcribe"
    f = _file(tmp, b"wxyz")
    msg = {"messageId": "m-1", "role": "ROLE_USER", "parts": [{"data": {"path": str(f), "lang": "en", "context": "podcast"}}],
           "metadata": {"caller": "youtube-agent"}}
    r1 = _rpc(c, "SendMessage", {"message": msg})["task"]
    r2 = _rpc(c, "SendMessage", {"message": msg})["task"]
    assert r1["id"] == r2["id"]
    assert r1["status"]["state"] == "TASK_STATE_WORKING"
    jid = r1["metadata"]["jobId"]
    j = wait_status(c, jid, "done")
    assert j["origin"] == "a2a" and j["caller"] == "youtube-agent"      # 화면 목록에 A2A 로 보인다
    t = _rpc(c, "GetTask", {"id": r1["id"]})
    assert t["status"]["state"] == "TASK_STATE_COMPLETED"
    assert t["metadata"]["reviewUrl"].endswith(f"/#job={jid}")
    assert t["artifacts"][0]["parts"][1]["data"]["segments"][0]["speaker"] == "1"
    # 사람이 화면에서 고친 것이 A2A 결과에도
    c.patch(f"/api/jobs/{jid}/segments/0", json={"text": "고친 문장."})
    assert "고친 문장." in _rpc(c, "GetTask", {"id": r1["id"]})["artifacts"][0]["parts"][0]["text"]
    # raw 파일 part + 취소
    eng.block.set()
    raw = {"messageId": "m-2", "role": "ROLE_USER", "parts": [{"raw": base64.b64encode(b"data").decode(), "filename": "c.wav"}]}
    t2 = _rpc(c, "SendMessage", {"message": raw})["task"]
    wait_status(c, t2["metadata"]["jobId"], "running")
    assert _rpc(c, "CancelTask", {"id": t2["id"]})["id"] == t2["id"]
    wait_status(c, t2["metadata"]["jobId"], "canceled")
    assert _rpc(c, "GetTask", {"id": t2["id"]})["status"]["state"] == "TASK_STATE_CANCELED"
    eng.block.clear()
    # 형식 오류: 파일 없음 / 없는 경로 / http url
    assert _rpc(c, "SendMessage", {"message": {"messageId": "m-3", "role": "ROLE_USER", "parts": [{"text": "hi"}]}}, err=True)["code"] == -32602
    assert _rpc(c, "SendMessage", {"message": {"role": "ROLE_USER", "parts": [{"data": {"path": "/nope.wav"}}]}}, err=True)["code"] == -32602
    assert _rpc(c, "SendMessage", {"message": {"role": "ROLE_USER", "parts": [{"url": "ftp://x/y.mp3"}]}}, err=True)["code"] == -32005
    assert _rpc(c, "GetTask", {"id": "없음"}, err=True)["code"] == -32001
    assert len(_rpc(c, "ListTasks", {})["tasks"]) >= 2


def test_live_mic_lines_then_final(env):
    c, _, _ = env
    job = c.post("/api/live", json={"source": "mic", "lang": "ko"}).json()
    assert job["status"] == "live"
    with c.websocket_connect(f"/api/live/{job['id']}/ws") as ws:
        ws.send_bytes(b"\0" * 64000)                       # 2초
        end = time.time() + 5
        while len(c.get(f"/api/jobs/{job['id']}").json()["segments"]) < 2:
            assert time.time() < end
            time.sleep(0.03)
        ws.send_text("stop")
    j = wait_status(c, job["id"], "done")
    assert j["kind"] == "live" and [s["speaker"] for s in j["segments"]] == ["1", "0"]   # 최종본이 실시간 줄을 대신함


def test_live_stop_without_final_and_empty(env):
    c, _, _ = env
    job = c.post("/api/live", json={"source": "mic"}).json()
    with c.websocket_connect(f"/api/live/{job['id']}/ws") as ws:
        ws.send_bytes(b"\0" * 32000)
        time.sleep(0.2)
        c.post(f"/api/live/{job['id']}/stop?final=false")
    j = wait_status(c, job["id"], "done")
    assert "최종본 없이" in j["warning"]
    empty = c.post("/api/live", json={"source": "mic"}).json()
    c.post(f"/api/live/{empty['id']}/stop")
    assert wait_status(c, empty["id"], "failed")["error"] == "들어온 소리가 없습니다"
    assert c.post("/api/live", json={"source": "stream"}).status_code == 400


def _file(tmp: Path, body: bytes) -> Path:
    p = tmp / f"in-{time.time_ns()}.wav"
    p.write_bytes(body)
    return p


def _rpc(c, method, params, err=False):
    r = c.post("/a2a", json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).json()
    if err:
        assert "error" in r, r
        return r["error"]
    assert "result" in r, r
    return r["result"]


def test_url_job_fetch_title_context_and_failures(env):
    c, eng, _ = env
    assert c.post("/api/jobs/url", json={"url": "ftp://x/y.mp3"}).status_code == 422 or \
        c.post("/api/jobs/url", json={"url": "ftp://x/y.mp3"}).status_code == 400
    job = c.post("/api/jobs/url", json={"url": "https://www.youtube.com/watch?v=DHRm_2A2Qag"}).json()
    assert job["title"].startswith("https://") and job["source_url"]
    j = wait_status(c, job["id"], "done")
    assert j["title"] == "DATING | Intermediate english podcast"            # 영상 제목으로 바뀜
    assert j["context"].startswith("DATING") and "English Cactus" in j["context"]   # 맥락 자동 채움
    assert ("transcribe", True, j["context"]) in eng.calls
    assert c.get(f"/api/jobs/{job['id']}/audio").status_code == 200
    # 사람이 준 제목·맥락은 그대로
    k = c.post("/api/jobs/url", json={"url": "https://youtu.be/x", "title": "내 제목", "context": "용어"}).json()
    k = wait_status(c, k["id"], "done")
    assert k["title"] == "내 제목" and k["context"] == "용어"
    # 받기 실패 → 실패 작업 (원인 글 그대로)
    bad = c.post("/api/jobs/url", json={"url": "https://bad.example/v"}).json()
    b = wait_status(c, bad["id"], "failed")
    assert b["error_kind"] == "fetch" and "404" in b["error"]
    # 받는 중 취소
    eng.block_fetch.set()
    w = c.post("/api/jobs/url", json={"url": "https://www.youtube.com/watch?v=slow"}).json()
    wait_status(c, w["id"], "running")
    assert c.get(f"/api/jobs/{w['id']}").json()["stage"] == "소리 받는 중"
    c.post(f"/api/jobs/{w['id']}/cancel")
    wait_status(c, w["id"], "canceled")
    eng.block_fetch.clear()
    # 다시 받아 적기는 다시 받지 않는다 (소리가 이미 있음)
    n = sum(1 for x in eng.calls if x[0] == "fetch")
    c.post(f"/api/jobs/{job['id']}/retranscribe")
    wait_status(c, job["id"], "done")
    assert sum(1 for x in eng.calls if x[0] == "fetch") == n


def test_a2a_url_part_and_data_url(env):
    c, _, _ = env
    t = _rpc(c, "SendMessage", {"message": {"messageId": "u-1", "role": "ROLE_USER", "parts": [{"url": "https://www.youtube.com/watch?v=DHRm_2A2Qag"}]}})["task"]
    j = wait_status(c, t["metadata"]["jobId"], "done")
    assert j["origin"] == "a2a" and j["title"].startswith("DATING")
    t2 = _rpc(c, "SendMessage", {"message": {"messageId": "u-2", "role": "ROLE_USER", "parts": [{"data": {"url": "https://youtu.be/abc", "lang": "en"}}]}})["task"]
    assert wait_status(c, t2["metadata"]["jobId"], "done")["lang"] == "en"
    assert _rpc(c, "SendMessage", {"message": {"role": "ROLE_USER", "parts": [{"url": "ftp://x/y"}]}}, err=True)["code"] == -32005


def test_restart_refetches_only_when_audio_missing(tmp_path):
    eng = FakeEngine()
    svc = Service(tmp_path / "d", eng, start_worker=False)
    j = svc.create_url_job("https://www.youtube.com/watch?v=DHRm_2A2Qag")
    svc.store.update_job(j["id"], status="running")                          # 받기 전에 꺼진 것처럼
    with TestClient(create_app(tmp_path / "d", engine=eng)) as c:
        assert wait_status(c, j["id"], "done")["title"].startswith("DATING")
    assert sum(1 for x in eng.calls if x[0] == "fetch") == 1
