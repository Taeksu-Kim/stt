"""python -m stt serve [--data data] [--port 8773]        화면 + 제품 API + A2A (127.0.0.1 고정)
python -m stt transcribe <파일|주소> ... [선택지]        서버 없이 받아 적기 (cli.py)
python -m stt live <ffmpeg 입력 | -> [선택지]            서버 없이 실시간 받아 적기 (cli.py)

Windows 브라우저에서도 http://127.0.0.1:<port> ("localhost"는 IPv6로 먼저 가서 시간 초과가 날 수 있음).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HOST = "127.0.0.1"  # 고정: 녹음·받아 적은 글은 이 PC 밖으로 열지 않는다
DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data"


def main(argv: list[str] | None = None) -> int:
    from stt import cli

    ap = argparse.ArgumentParser(prog="stt")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="화면 + 제품 API + A2A")
    s.add_argument("--data", default=str(DEFAULT_DATA))
    s.add_argument("--port", type=int, default=8773)
    s.add_argument("--allow-download", action="store_true")
    t = sub.add_parser("transcribe", help="파일·주소 받아 적기 (서버 없이)")
    t.add_argument("inputs", nargs="+", help="소리·영상 파일 또는 http(s) 주소 (유튜브 등 — 소리만 받음)")
    cli.add_common(t)
    lv = sub.add_parser("live", help="실시간 받아 적기 (서버 없이)")
    lv.add_argument("source", help="ffmpeg 입력(방송 주소·파일·장치) 또는 '-'(표준 입력 PCM16 16kHz mono)")
    lv.add_argument("--realtime", action="store_true", help="파일을 실제 재생 속도로 흘림 (시험용)")
    lv.add_argument("--duration", type=float, help="이 초만큼만 듣고 끝냄")
    lv.add_argument("--no-final", action="store_true", help="끝난 뒤 녹음본 최종 받아 적기를 건너뜀")
    cli.add_common(lv)
    a = ap.parse_args(argv)
    if a.cmd == "transcribe":
        return cli.run_transcribe(a)
    if a.cmd == "live":
        return cli.run_live(a)

    import uvicorn

    from stt.app import create_app
    from stt.engine import Engine

    url = f"http://{HOST}:{a.port}"
    print(f"데이터: {a.data}\n열기: {url}\nA2A 카드: {url}/.well-known/agent-card.json")
    uvicorn.run(create_app(a.data, engine=Engine(allow_download=a.allow_download), self_url=url), host=HOST, port=a.port,
                log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
