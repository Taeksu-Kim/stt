# stt — 프로젝트 규칙

공통 규칙은 상위 `../CLAUDE.md`. 사용법은 `README.md`, 화면 콘셉트는 `docs/design/concept.md`, 결정·실패 기록은
`.claude/skills/stt/NOTES.md` (스킬과 같이 자란 기록 — 엔진 결정 D1~D11, 실패 F1~F6). 쓰는 법은 같은 폴더 `SKILL.md`.

- 층: 엔진 `engine.py`(단계: select·fetch·prepare·diarize·transcribe·live) → 서비스 `service.py`(상태의 유일한 주인, SQLite `data/stt.db`)
  → 제품 API `app.py` + A2A `a2a_tasks.py`(같은 서비스 함수). CLI `cli.py` 는 서버 없이 엔진만 쓴다.
- 모델·환경 경로는 `stt/models.json` 한 곳 (`STT_MODELS` 로 바꿈). 코드에 이 PC의 경로를 새로 넣지 않는다.
  엔진 프로세스는 등록부의 `python`(voice-eval) 으로 따로 띄운다 — 서버 환경(stock-dataset)에 torch·모델 패키지를 넣지 않는다.
- 인식은 15초 조각째로, 쉼(VAD)은 문장 시각 배분에만 (NOTES D5·D7 — 쉼에서 잘라 인식하면 단어 오류가 늘었다).
  조각 길이를 바꾸면 같은 한국어 샘플로 맞고 틀린 단어를 다시 대 본다.
- 밖으로 나가는 단계는 주소 받기(`Engine.fetch`, yt-dlp) 하나뿐. 받아 적기·화자 분리는 이 PC 안에서만 한다.
- 녹음·받아 적은 글은 `data/` 에만 (git 제외). 127.0.0.1 고정. 캡처·문서 예시는 공개된 유튜브 음성만 (README 캡처 = `tools/browser_check.py --shots`).
- 화면을 고치면 `ui-design` 스킬부터. 고친 뒤 `python tools/browser_check.py ...` (실제 엔진 서버에 대고) 와 `pytest` (모델만 가짜) 둘 다.
- A2A 결과에는 `metadata.reviewUrl`(= 화면 `#job=<id>`). 화면에서 사람이 고친 글이 A2A 결과에도 그대로 나간다.
