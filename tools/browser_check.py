"""실제 브라우저로 화면 과업을 눌러 본다 (CLAUDE.md 완료 확인). 실제 엔진이 켜진 서버에 대고 돈다.

    python tools/browser_check.py --base http://127.0.0.1:8774 --en <영어.m4a> --ko <한국어.m4a> --short <20초.m4a>
                                  --mic <말소리.wav> --silent <무음.wav> --shots <캡처 폴더>

순서: A2A 로 한국어 맡김 + 같은 요청 2회 / 화면에서 영어·무음 올리기 / 목록·A2A 표시 / 재생 / 화자 이름 / 고치기(한글 IME Enter) /
여기까지 확인 / 검색 → 이동 / 내려받기 / 구간 다시 받아 적기 / 올리고 바로 취소 / 실시간(가짜 마이크 = 실제 말소리 파일) → 최종본.
캡처: 단계마다 <shots>/<단계>.png (공개 파일만 띄운 서버에서). 실패하면 어느 단계인지 내고 종료 코드 1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
from playwright.sync_api import expect, sync_playwright

ok_count, fails = 0, []


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok_count
    if cond:
        ok_count += 1
        print(f"  ok  {name}")
    else:
        fails.append(f"{name} {detail}")
        print(f"  FAIL {name} {detail}")


def wait_job(base: str, jid: str, want: tuple[str, ...], timeout: float = 240) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        j = httpx.get(f"{base}/api/jobs/{jid}").json()
        if j["status"] in want:
            return j
        time.sleep(1)
    raise TimeoutError(f"{jid} {j['status']}")


def rpc(base: str, method: str, params: dict) -> dict:
    return httpx.post(f"{base}/a2a", json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, timeout=60).json()


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("base", "en", "ko", "short", "mic", "silent", "shots"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--url", help="주소로 받아 적기까지 볼 때 (유튜브 등 — 소리만 받음, 인터넷 필요)")
    a = ap.parse_args()
    base, shots = a.base.rstrip("/"), Path(a.shots)
    shots.mkdir(parents=True, exist_ok=True)

    print("1. A2A 로 맡기기 (같은 요청 2회)")
    msg = {"messageId": "check-ko-1", "role": "ROLE_USER", "metadata": {"caller": "youtube-agent"},
           "parts": [{"data": {"path": str(Path(a.ko).resolve()), "lang": "ko", "title": "대기업 회장님이 생각하는 한국 경제의 미래",
                               "context": "김지윤의 지식Play, SK, 최태원"}}]}
    t1, t2 = rpc(base, "SendMessage", {"message": msg})["result"]["task"], rpc(base, "SendMessage", {"message": msg})["result"]["task"]
    check("같은 messageId → 같은 작업", t1["id"] == t2["id"])
    ko_job = t1["metadata"]["jobId"]

    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                                    f"--use-file-for-fake-audio-capture={Path(a.mic).resolve()}", "--autoplay-policy=no-user-gesture-required"])
        pg = b.new_page(viewport={"width": 1280, "height": 860})
        errors: list[str] = []
        pg.on("console", lambda m: m.type == "error" and "favicon" not in m.text and errors.append(m.text))
        pg.on("pageerror", lambda e: errors.append(str(e)))

        print("2. 화면에서 올리기 (영어 + 무음)")
        pg.goto(base + "/")
        pg.wait_for_timeout(800)
        pg.click("#bUp")
        pg.set_input_files("#fileIn", [a.en, a.silent])
        pg.select_option("#uLang", "")
        pg.fill("#uCtx", "English Cactus podcast, Anna, Brian, Sophie")
        pg.screenshot(path=str(shots / "02-upload-dialog.png"))
        pg.click("#upGo")
        pg.wait_for_timeout(1500)
        pg.screenshot(path=str(shots / "03-running.png"))
        js = httpx.get(base + "/api/jobs").json()
        en_job = next(j["id"] for j in js if j["audio_name"] == Path(a.en).name)
        silent_job = next(j["id"] for j in js if j["audio_name"] == Path(a.silent).name)
        check("올린 2개가 줄에 들어감", all(j["status"] in ("queued", "running", "done") for j in js if j["id"] in (en_job, silent_job)))
        for jid in (ko_job, en_job, silent_job):
            wait_job(base, jid, ("done", "failed"))
        pg.wait_for_timeout(1200)

        print("3. 목록 — A2A 표시, 순서")
        tags = pg.locator(".job .tag").all_inner_texts()
        check("A2A 작업이 목록에 A2A 로 보임", "A2A" in tags, str(tags))
        t = rpc(base, "GetTask", {"id": t1["id"]})["result"]
        check("A2A 결과: COMPLETED + reviewUrl", t["status"]["state"] == "TASK_STATE_COMPLETED" and t["metadata"]["reviewUrl"].endswith(ko_job))

        print("4. 영어 작업 — 재생·화자 이름·고치기·확인")
        pg.click(f'[data-job="{en_job}"]')
        pg.wait_for_selector(".segs tr")
        pg.screenshot(path=str(shots / "04-done.png"))
        row = pg.locator('.segs tr[data-idx="5"]')
        row.click()
        pg.wait_for_timeout(1500)
        cur = pg.evaluate("document.getElementById('au').currentTime")
        seg5 = httpx.get(f"{base}/api/jobs/{en_job}").json()["segments"][5]
        check("문장 누르면 그 시각부터 재생", seg5["start"] - 0.2 <= cur <= seg5["start"] + 3, f"{cur:.1f} vs {seg5['start']}")
        check("재생 중 그 줄이 현재 줄", "now" in (row.get_attribute("class") or ""))
        pg.mouse.move(5, 5)                               # 마우스를 치워도
        check("현재 줄은 동작이 늘 보임", row.locator('[data-act="edit"]').is_visible())
        check("다른 줄은 동작이 숨음", not pg.locator('.segs tr[data-idx="9"] [data-act="edit"]').is_visible())
        pg.click('[data-act="more"]')
        check("⋯ 메뉴에 위험 동작 (다시 받아 적기·지우기)", pg.locator('#moreMenu [data-act="retranscribe"]').is_visible() and pg.locator('#moreMenu [data-act="delete"]').is_visible())
        pg.keyboard.press("Escape")
        check("Esc 로 메뉴 닫힘", not pg.locator('#moreMenu [data-act="delete"]').is_visible())
        pg.click("#pb")                                   # 멈춤
        pg.locator(".lb .nm").first.click()
        inp = pg.locator("[data-spk-in]").first
        inp.fill("Anna")
        inp.press("Enter")
        pg.wait_for_timeout(600)
        check("화자 이름 붙이기", "Anna" in pg.locator(".lb .nm").first.inner_text())
        r3 = pg.locator('.segs tr[data-idx="3"]')
        r3.hover()
        r3.locator('[data-act="edit"]').click()
        ta = pg.locator("#edT")
        ta.fill("And my name is Brian.")
        ta.press("Enter")
        pg.wait_for_timeout(700)
        s3 = httpx.get(f"{base}/api/jobs/{en_job}").json()["segments"][3]
        check("고치기 저장 (Enter)", s3["text"] == "And my name is Brian." and s3["checked"] == 1, s3["text"])
        check("고친 곳 표시 (del/ins + 트랙 눈금)", pg.locator('.segs tr[data-idx="3"] ins').count() == 1 and pg.locator(".et").count() >= 1)
        # 한글 입력 중 Enter 는 저장하지 않는다 (조합 중 keyCode 229)
        r4 = pg.locator('.segs tr[data-idx="4"]')
        r4.hover()
        r4.locator('[data-act="edit"]').click()
        pg.evaluate("""(()=>{const t=document.getElementById('edT');t.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',keyCode:229,isComposing:true,bubbles:true}))})()""")
        pg.wait_for_timeout(300)
        check("한글 조합 중 Enter 는 저장 안 함", pg.locator("#edT").count() == 1)
        pg.locator("#edT").press("Escape")
        r11 = pg.locator('.segs tr[data-idx="11"]')
        r11.hover()
        r11.locator('[data-act="checkUpTo"]').click()
        pg.wait_for_timeout(700)
        jd = httpx.get(f"{base}/api/jobs/{en_job}").json()
        check("여기까지 확인 → 12문장", jd["n_checked"] == 12, str(jd["n_checked"]))
        pg.locator('.segs tr[data-idx="12"] .chk').click()
        pg.wait_for_timeout(500)
        check("확인 한 줄 누르기", httpx.get(f"{base}/api/jobs/{en_job}").json()["n_checked"] == 13)
        pg.locator('.lb .nm').nth(1).click()
        pg.locator("[data-spk-in]").first.fill("Brian")
        pg.locator("[data-spk-in]").first.press("Enter")
        pg.wait_for_timeout(500)
        pg.locator("#scroll").evaluate("e => e.scrollTop = 0")
        pg.locator('.segs tr[data-idx="5"]').click()
        pg.wait_for_timeout(900)
        pg.click("#pb")
        pg.wait_for_timeout(300)
        pg.screenshot(path=str(shots / "05-reviewing.png"))
        txt = httpx.get(f"{base}/api/jobs/{en_job}/export?fmt=txt").text
        check("내려받기 txt 에 화자 이름·고친 글", "[Anna]" in txt and "And my name is Brian." in txt)
        check("내려받기 srt", "-->" in httpx.get(f"{base}/api/jobs/{en_job}/export?fmt=srt").text)

        print("4-1. 확인 안 한 문장 거르기 · 다음 확인 안 한 줄")
        pg.click('[data-filter="todo"]')
        pg.wait_for_timeout(300)
        jd = httpx.get(f"{base}/api/jobs/{en_job}").json()
        todo = sum(1 for s in jd["segments"] if not s["checked"])
        check("확인 안 함 탭 = 확인 안 한 문장 수", pg.locator(".segs tbody tr").count() == todo, f"{pg.locator('.segs tbody tr').count()} vs {todo}")
        pg.click('[data-filter="all"]')
        pg.locator("body").press("n")
        pg.wait_for_timeout(400)
        first_todo = next(s["idx"] for s in jd["segments"] if not s["checked"])
        check("N = 다음 확인 안 한 줄로", "flash" in (pg.locator(f'.segs tr[data-idx="{first_todo}"]').get_attribute("class") or ""))
        print("5. 검색 → 이동")
        pg.fill("#q", "Cactus")
        pg.wait_for_selector(".res")
        pg.screenshot(path=str(shots / "06-search.png"))
        pg.locator(".res").first.click()
        pg.wait_for_selector(".segs tr.hit")
        check("검색 결과 누르면 그 문장으로", pg.locator(".segs tr.hit").count() == 1)
        pg.click("#qClear")

        print("6. 구간 다시 받아 적기")
        r7 = pg.locator('.segs tr[data-idx="7"]')
        r7.hover()
        r7.locator('[data-act="redo"]').click()
        end = time.time() + 120
        while time.time() < end and httpx.get(f"{base}/api/jobs/{en_job}").json()["redoing"]:
            time.sleep(1)
        check("구간 다시 받아 적기 끝남", not httpx.get(f"{base}/api/jobs/{en_job}").json()["redoing"])

        print("7. 한국어(A2A) · 무음 화면")
        pg.click(f'[data-job="{ko_job}"]')
        pg.wait_for_selector(".segs tr")
        pg.screenshot(path=str(shots / "07-a2a-korean.png"))
        pg.click(f'[data-job="{silent_job}"]')
        pg.wait_for_timeout(500)
        check("무음 = 말소리 없음 안내", "말소리를 찾지 못했습니다" in pg.inner_text("main"))
        pg.screenshot(path=str(shots / "08-silent.png"))

        print("8. 올리고 바로 취소")
        pg.click("#bUp")
        pg.set_input_files("#fileIn", [a.short])
        pg.click("#upGo")
        pg.wait_for_selector('[data-act="cancel"]')
        pg.click('[data-act="cancel"]')
        js = httpx.get(base + "/api/jobs").json()
        short_job = next(j["id"] for j in js if j["audio_name"] == Path(a.short).name)
        check("취소", wait_job(base, short_job, ("canceled", "done"), 60)["status"] == "canceled")
        pg.wait_for_timeout(600)
        pg.screenshot(path=str(shots / "09-canceled.png"))

        print("9. 실시간 (가짜 마이크 = 실제 말소리)")
        pg.click("#bLive")
        pg.fill("#lTitle", "실시간 마이크 점검")
        pg.screenshot(path=str(shots / "10-live-dialog.png"))
        pg.click("#liveGo")
        pg.wait_for_selector(".bar2.live")
        end = time.time() + 90
        live_id = pg.evaluate("location.hash").split("=")[1]
        while time.time() < end and len(httpx.get(f"{base}/api/jobs/{live_id}").json()["segments"]) < 2:
            time.sleep(1)
        pg.wait_for_timeout(800)
        n_live = len(httpx.get(f"{base}/api/jobs/{live_id}").json()["segments"])
        check("실시간 줄이 쌓임", n_live >= 2, str(n_live))
        pg.screenshot(path=str(shots / "11-live.png"))
        pg.click('[data-act="stop"]')
        j = wait_job(base, live_id, ("done", "failed"), 180)
        check("끝내면 최종본 (화자 포함)", j["status"] == "done" and any(s["speaker"] is not None for s in j["segments"]), j["status"])
        pg.wait_for_timeout(1200)
        pg.screenshot(path=str(shots / "12-live-final.png"))

        print("10. 화자 합치기 (실시간 최종본)")
        pg.reload()
        pg.wait_for_selector(".lb .nm")
        names = pg.locator(".lb .nm").all_inner_texts()
        if len(names) >= 2:
            before = {s["speaker"] for s in httpx.get(f"{base}/api/jobs/{live_id}").json()["segments"]}
            pg.locator(".lb .nm").nth(1).click()
            pg.wait_for_selector(".spop")
            pg.screenshot(path=str(shots / "13-speaker-pop.png"))
            moved = pg.locator(".lb .nm").nth(1).get_attribute("data-spk")
            pg.locator(".spop [data-merge]").first.click()
            pg.click("#dcGo")
            pg.wait_for_timeout(800)
            spk = {s["speaker"] for s in httpx.get(f"{base}/api/jobs/{live_id}").json()["segments"]}
            check("화자 합치기 → 합친 화자가 사라지고 하나 줄어듦", moved not in spk and len(spk) == len(before) - 1, f"{sorted(before)} → {sorted(spk)}")
        else:
            check("화자 합치기 (화자 2명 이상 필요)", False, str(names))
        if a.url:
            print("11. 주소로 받아 적기 (붙여 넣기 → 대화창)")
            pg.evaluate("t => { const d = new DataTransfer(); d.setData('text/plain', t); document.body.dispatchEvent(new ClipboardEvent('paste', {clipboardData: d, bubbles: true})); }", a.url)
            pg.wait_for_selector("#dUp[open]")
            check("주소 붙여 넣기 → 주소 칸이 채워진 넣기 창", pg.input_value("#uUrl") == a.url and pg.locator("#uUrlF").is_visible())
            pg.screenshot(path=str(shots / "14-url-dialog.png"))
            pg.click("#upGo")
            pg.wait_for_timeout(1500)
            url_job = pg.evaluate("location.hash").split("=")[1]
            pg.screenshot(path=str(shots / "15-url-fetching.png"))
            j = wait_job(base, url_job, ("done", "failed"), 420)
            check("주소 → 받아 적기 끝 (제목·맥락은 영상에서)", j["status"] == "done" and not j["title"].startswith("http") and bool(j["context"]),
                  f'{j["status"]} {j["title"][:40]} {(j["error"] or "")[:80]}')
            pg.wait_for_timeout(1200)
            check("원본 영상 링크", pg.locator(".meta .srcl").count() == 1)
            pg.screenshot(path=str(shots / "16-url-done.png"))
        check("콘솔 오류 0건", not errors, json.dumps(errors, ensure_ascii=False)[:400])
        b.close()
    print(f"\n통과 {ok_count} / 실패 {len(fails)}")
    for f in fails:
        print("  -", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
