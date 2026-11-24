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
                   "channel": "English Cactus Podcast", "upload_date": "20240501", "extractor": "youtube",
                   "webpage_url": "https://www.youtube.com/watch?v=DHRm_2A2Qag",
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


# ---------------- 같은 영상 다시 오면 (NOTES D12) ----------------
def _url(c, url, **body):
    r = c.post("/api/jobs/url", json={"url": url, **body})
    assert r.status_code == 200, r.text
    return r.json()


def _fetches(eng):
    return sum(1 for x in eng.calls if x[0] == "fetch")


def test_same_video_url_variants_reused_and_force(env):
    c, eng, _ = env
    a = _url(c, "https://www.youtube.com/watch?v=dhojLjm7c9w")
    assert a["reused"] is False and a["source_key"] == "youtube:dhojLjm7c9w"
    j = wait_status(c, a["id"], "done")
    # 영상 정보 칸이 채워짐 (yt-dlp -J 메타에서)
    assert j["video"] == {"video_id": "DHRm_2A2Qag", "channel": "English Cactus Podcast", "upload_date": "2024-05-01",
                          "site": "youtube", "webpage_url": "https://www.youtube.com/watch?v=DHRm_2A2Qag", "duration": 12.0}
    for v in ("https://youtu.be/dhojLjm7c9w?t=10", "https://m.youtube.com/watch?v=dhojLjm7c9w&t=10s&list=PLx",
              "https://www.youtube.com/shorts/dhojLjm7c9w"):
        b = _url(c, v)
        assert b["reused"] is True and b["id"] == a["id"] and b["status"] == "done", v
    assert _fetches(eng) == 1 and len(c.get("/api/jobs").json()) == 1
    # 일부러 새로: force
    f = _url(c, "https://youtu.be/dhojLjm7c9w", force=True)
    assert f["reused"] is False and f["id"] != a["id"]
    wait_status(c, f["id"], "done")
    assert _fetches(eng) == 2
    # 둘 다 끝난 뒤 다시 오면 가장 최근 것
    assert _url(c, "https://www.youtube.com/watch?v=dhojLjm7c9w")["id"] == f["id"]
    # 다른 주소는 주소 그대로가 키
    assert _url(c, "https://vimeo.com/123")["source_key"] == "https://vimeo.com/123"
    # 검색: 채널 이름으로 작업이 걸린다 (문장 없이 idx=None)
    hits = c.get("/api/search?q=Cactus").json()
    assert any(h["idx"] is None and h["job_id"] == f["id"] for h in hits)


def test_failed_or_canceled_only_makes_new_running_is_reused(env):
    c, eng, _ = env
    bad = _url(c, "https://bad.example/v")
    wait_status(c, bad["id"], "failed")
    again = _url(c, "https://bad.example/v")
    assert again["reused"] is False and again["id"] != bad["id"]
    wait_status(c, again["id"], "failed")
    # 받는 중 → 같은 작업 / 취소한 것만 있으면 새로
    eng.block_fetch.set()
    w = _url(c, "https://www.youtube.com/watch?v=slow1")
    wait_status(c, w["id"], "running")
    r = _url(c, "https://youtu.be/slow1")
    assert r["reused"] is True and r["id"] == w["id"] and r["status"] == "running"
    q = _url(c, "https://www.youtube.com/watch?v=slow2")       # 줄에서 기다리는 중 → 같은 작업
    assert _url(c, "https://youtu.be/slow2")["id"] == q["id"]
    c.post(f"/api/jobs/{w['id']}/cancel")
    c.post(f"/api/jobs/{q['id']}/cancel")
    wait_status(c, w["id"], "canceled")
    eng.block_fetch.clear()
    n = _url(c, "https://youtu.be/slow1")
    assert n["reused"] is False and n["id"] != w["id"]
    wait_status(c, n["id"], "done")


def test_language_and_speakers_decide_reuse(env):
    c, _, _ = env
    a = _url(c, "https://www.youtube.com/watch?v=lang1")              # 자동 → 알아낸 언어 en (가짜 엔진)
    wait_status(c, a["id"], "done")
    assert _url(c, "https://youtu.be/lang1", lang="en")["id"] == a["id"]      # 알아낸 언어와 같음 → 재사용
    ko = _url(c, "https://youtu.be/lang1", lang="ko")                         # 다르게 명시 → 새로
    assert ko["reused"] is False and ko["id"] != a["id"] and ko["lang"] == "ko"
    wait_status(c, ko["id"], "done")
    assert _url(c, "https://youtu.be/lang1", lang="ko")["id"] == ko["id"]
    # 화자 분리 없이 한 작업 → 화자 분리를 원하면 새로, 원하지 않으면 재사용
    s0 = _url(c, "https://www.youtube.com/watch?v=spk1", speakers=False)
    wait_status(c, s0["id"], "done")
    assert _url(c, "https://youtu.be/spk1", speakers=False)["id"] == s0["id"]
    s1 = _url(c, "https://youtu.be/spk1")
    assert s1["reused"] is False and s1["id"] != s0["id"]
    wait_status(c, s1["id"], "done")
    assert _url(c, "https://youtu.be/spk1", speakers=False)["id"] == s1["id"]      # 화자 분리 한 것은 없이 원해도 재사용


def test_a2a_same_video_two_messages_same_job(env):
    c, eng, _ = env
    m1 = {"messageId": "v-1", "role": "ROLE_USER", "parts": [{"data": {"url": "https://www.youtube.com/watch?v=dhojLjm7c9w"}}],
          "metadata": {"caller": "video_reader"}}
    t1 = _rpc(c, "SendMessage", {"message": m1})["task"]
    assert t1["metadata"]["reused"] is False
    wait_status(c, t1["metadata"]["jobId"], "done")
    m2 = {"messageId": "v-2", "role": "ROLE_USER", "parts": [{"url": "https://youtu.be/dhojLjm7c9w?t=10"}]}
    t2 = _rpc(c, "SendMessage", {"message": m2})["task"]
    assert t2["id"] != t1["id"] and t2["metadata"]["jobId"] == t1["metadata"]["jobId"]
    assert t2["metadata"]["reused"] is True and t2["status"]["state"] == "TASK_STATE_COMPLETED"    # 끝난 결과 그대로
    meta = t2["artifacts"][0]["parts"][1]["data"]["meta"]
    assert meta["video"]["channel"] == "English Cactus Podcast" and meta["video"]["upload_date"] == "2024-05-01"
    assert t2["metadata"]["video"]["video_id"] == "DHRm_2A2Qag"
    assert _fetches(eng) == 1
    # 같은 messageId 다시 → 그 A2A 작업 그대로 / force → 새 작업
    assert _rpc(c, "SendMessage", {"message": m2})["task"]["id"] == t2["id"]
    m3 = {"messageId": "v-3", "role": "ROLE_USER", "parts": [{"data": {"url": "https://youtu.be/dhojLjm7c9w", "force": True}}]}
    t3 = _rpc(c, "SendMessage", {"message": m3})["task"]
    assert t3["metadata"]["jobId"] != t1["metadata"]["jobId"] and t3["metadata"]["reused"] is False
    wait_status(c, t3["metadata"]["jobId"], "done")


OLD_SCHEMA = """
CREATE TABLE jobs (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, kind TEXT NOT NULL, origin TEXT NOT NULL, caller TEXT,
  status TEXT NOT NULL, stage TEXT, audio_path TEXT, audio_name TEXT, duration REAL,
  lang TEXT, context TEXT, num_speakers INTEGER, speakers_on INTEGER NOT NULL DEFAULT 1, source_url TEXT,
  engine TEXT, model TEXT, diarizer TEXT, language TEXT, warning TEXT, error TEXT, error_kind TEXT,
  timings TEXT, created TEXT NOT NULL, updated TEXT NOT NULL, started TEXT, finished TEXT
);
CREATE TABLE segments (
  job_id TEXT NOT NULL, idx INTEGER NOT NULL, start REAL NOT NULL, end REAL NOT NULL, speaker TEXT,
  text TEXT NOT NULL, asr_text TEXT NOT NULL, timing TEXT, checked INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (job_id, idx)
);
CREATE TABLE speakers (job_id TEXT NOT NULL, label TEXT NOT NULL, name TEXT NOT NULL, PRIMARY KEY (job_id, label));
CREATE TABLE a2a_tasks (
  id TEXT PRIMARY KEY, context_id TEXT NOT NULL, message_id TEXT UNIQUE, job_id TEXT, state TEXT, error TEXT, created TEXT NOT NULL
);
INSERT INTO jobs (id, title, kind, origin, status, speakers_on, source_url, lang, language, created, updated)
  VALUES ('old1', '옛 영상', 'file', 'screen', 'done', 1, 'https://www.youtube.com/watch?v=oldVid&t=3', NULL, 'ko',
          '2026-11-20 10:00:00', '2026-11-20 10:00:00');
INSERT INTO segments VALUES ('old1', 0, 0.0, 2.0, '0', '옛 문장', '옛 문장', 'piece', 0);
INSERT INTO a2a_tasks VALUES ('t-old', 'ctx', 'm-old', 'old1', NULL, NULL, '2026-11-20 10:00:00');
"""


def test_old_schema_db_migrates_and_reuses(tmp_path):
    import sqlite3
    d = tmp_path / "d"
    d.mkdir()
    db = sqlite3.connect(str(d / "stt.db"))
    db.executescript(OLD_SCHEMA)
    db.close()
    eng = FakeEngine()
    with TestClient(create_app(d, engine=eng)) as c:
        cols = {r[1] for r in sqlite3.connect(str(d / "stt.db")).execute("PRAGMA table_info(jobs)")}
        assert {"source_key", "video_id", "channel", "upload_date", "site", "webpage_url"} <= cols
        old = c.get("/api/jobs/old1").json()
        assert old["source_key"] == "youtube:oldVid" and old["video"] is None and old["updated"] == "2026-11-20 10:00:00"
        r = _url(c, "https://youtu.be/oldVid")                   # 옛 작업도 같은 영상으로 찾힘
        assert r["reused"] is True and r["id"] == "old1"
        assert _rpc(c, "GetTask", {"id": "t-old"})["status"]["state"] == "TASK_STATE_COMPLETED"   # 옛 A2A 작업도 읽힘
        n = _url(c, "https://www.youtube.com/watch?v=newVid")
        assert wait_status(c, n["id"], "done")["video"]["channel"] == "English Cactus Podcast"
    assert _fetches(eng) == 1
    Service(d, eng, start_worker=False)                        # 두 번째 시작 — 이미 있는 칸은 그대로 (ALTER 다시 안 함)


def test_a2a_cancel_of_shared_job_only_detaches(env):
    """같은 영상을 두 요청이 함께 기다릴 때 한 요청의 취소가 다른 요청의 받아 적기를 끊지 않는다."""
    c, eng, _ = env
    eng.block_fetch.set()
    msg = lambda mid: {"message": {"messageId": mid, "role": "ROLE_USER", "parts": [{"url": "https://www.youtube.com/watch?v=shared01"}]}}
    t1 = _rpc(c, "SendMessage", msg("s-1"))["task"]
    t2 = _rpc(c, "SendMessage", msg("s-2"))["task"]
    jid = t1["metadata"]["jobId"]
    assert t2["metadata"]["jobId"] == jid
    wait_status(c, jid, "running")
    assert _rpc(c, "CancelTask", {"id": t2["id"]})["status"]["state"] == "TASK_STATE_CANCELED"
    assert c.get(f"/api/jobs/{jid}").json()["status"] == "running"          # 작업은 계속
    assert _rpc(c, "GetTask", {"id": t1["id"]})["status"]["state"] == "TASK_STATE_WORKING"
    # 남은 마지막 요청이 취소하면 작업도 취소
    _rpc(c, "CancelTask", {"id": t1["id"]})                                    # 작업을 멈추라는 요청 (멈추기까지는 잠깐)
    wait_status(c, jid, "canceled")
    assert _rpc(c, "GetTask", {"id": t1["id"]})["status"]["state"] == "TASK_STATE_CANCELED"
    eng.block_fetch.clear()
