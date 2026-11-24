---
name: stt
description: 음성·영상을 로컬 모델로 받아 적는다(STT·전사) — 파일·유튜브 등 영상 주소·실시간 입력(라이브 방송·스트림·다른 프로그램이 흘려 주는 PCM). 문장마다 시각과 화자가 붙은 글·자막을 내고, 사람이 화면에서 들으며 확인·고치게 한다. 엔진(Qwen3-ASR·Whisper)과 화자 분리(DiariZen)는 등록부에서 자동으로 고른다. "받아 적어 줘", "텍스트로 뽑아 줘", "유튜브 내용 텍스트화", "이 유튜브 받아 적어", "녹음 파일 글로", "자막 만들어", "srt", "누가 말했는지", "화자 구분", "전사해 줘", "실시간으로 받아 적어", "라이브 방송 받아 적기", 영상·팟캐스트·회의 녹음을 글로 바꿔야 할 때, 음성 모델 경로·엔진을 추가하거나 바꿀 때. 아닌 것 — 글을 음성으로 만드는 TTS, 받아 적은 글의 번역·요약(그건 받아 적은 뒤 LLM 일).
---

# stt — 로컬 음성 받아 적기 (시각·화자 기본)

이 스킬은 저장소 `stt`(https://github.com/Taeksu-Kim/stt) 안에 있다. **저장소 루트 = 이 스킬 폴더의 세 단계 위**다.
링크로 걸려 있을 수 있으니 실제 경로로 푼다: `ROOT="$(realpath <이 스킬 폴더>)/../../.."` → 명령은 모두 `cd "$ROOT"` 에서.
`python` 은 서버 환경의 파이썬이다 (README "처음 쓰는 사람". 이 PC의 값은 상위 작업 폴더 `CLAUDE.md`).

흐름: 16kHz 변환 → **화자 분리로 발화 구간** → 같은 화자의 1초 미만 쉼은 합치고 15초를 넘는 구간은 조용한 지점에서 자름 → 조각째 받아 적기
→ 조각 글을 문장으로 나누고 글자 비율을 말소리 시간(VAD, 쉼 제외)에 배분해 문장 시각 추정.
`timing`: `piece`(조각 경계 — 정확) / `estimated`(추정, Whisper 대비 평균 0.27초). 고른 엔진과 못 고른 이유는 `meta.selection`.

## 0. 먼저 — 엔진이 준비됐나
```bash
cd "$ROOT" && python -m stt transcribe <아무 소리 파일> --dry-run
```
`would_use`가 비었으면 `selection[].why_not`(모델 없음·환경 없음·GPU 여유 부족)을 사용자에게 그대로 보이고 README "처음 쓰는 사람"으로 안내한다.
모델은 기본 오프라인 — 처음 받을 때만 `--allow-download` (용량·시간이 드니 사용자에게 먼저 묻는다).

## 1. 어느 길로
| 상황 | 길 |
|---|---|
| 사람이 듣고 확인·고칠 결과, 긴 녹음, 다른 에이전트가 맡김 | **서버** (§3) — 결과와 함께 화면 링크(`#job=<id>`)를 준다 |
| 글·자막 파일만 바로 필요 | **CLI** (§2) |
| 마이크 | 서버 화면의 "실시간 받아 적기 → 이 브라우저 마이크" (CLI는 WSL에서 마이크를 못 잡는다) |

## 2. CLI (서버 없이)
```bash
cd "$ROOT"
python -m stt transcribe <파일|주소> [...] --context "<영상 제목·설명·주요 용어>" [--lang en] [--num-speakers N] [--out-dir <폴더>]
python -m stt live "<ffmpeg 입력 | ->" [--duration 600] [--realtime] [--no-final]
```
- `--context`는 넣는다 — 고유명사가 덜 틀린다. **주소**(유튜브 등)면 비워 둬도 영상 제목·채널·설명을 맥락으로 쓴다. 언어를 알면 `--lang ko|en|ja…`.
- 주소는 소리만 받는다 (25분 영상 23MB·11초, 2026-11-24 측정 — 병목 아님). 공식 자막이 있으면 받아 적지 않고 자막을 쓰는 판단은
  유튜브 분석 에이전트 몫이다 — 이 스킬은 받아 적기만.
- 화자 수는 확실히 알 때만 `--num-speakers` (함정 4). `--no-speakers`는 사용자가 속도를 원할 때만 (시각이 15초 조각 단위가 됨).
- 결과 `<이름>.stt/`: `segments.json`(meta + `[{start,end,speaker,text,timing}]`) · `transcript.txt`(화자 머리 + `[시:분:초] 글`) · `subtitle.srt`.
  같은 파일·같은 조건이면 `skipped` — 다시 하려면 `--force`.
- 실시간(`live`): 말이 0.6초 쉬면 끊어 한 줄 JSON을 내고, 끝(소스 끝·`--duration`·Ctrl+C)나면 녹음본으로 최종본(화자 포함)을 다시 만든다.
  유튜브 생방송은 `"$(yt-dlp -g -f bestaudio <URL>)"`로 스트림 주소를 얻어 넣는다 (생방송으로는 시험 안 함).

## 3. 서버 (화면 · 제품 API · A2A)
```bash
cd "$ROOT" && python -m stt serve          # http://127.0.0.1:8773 (이미 떠 있으면 그 서버를 쓴다)
```
- 사람에게: 화면 주소를 주고, 주소는 화면 아무 곳에나 붙여 넣으면 넣기 창이 열린다고 알린다.
- 제품 API: `POST /api/jobs`(파일 multipart) · `POST /api/jobs/url` `{url, lang?, context?}` → `GET /api/jobs/<id>` 의 `status`가 `done`이면
  `segments`. 내려받기 `GET /api/jobs/<id>/export?fmt=txt|srt|json`.
- A2A: `/.well-known/agent-card.json`, `POST /a2a` `SendMessage` — 파일 part / `file://` / `data.path` / https 주소(`url` part 또는 `data.url`).
  처음엔 WORKING → `GetTask`로 COMPLETED + artifact(글 + 구간) + `metadata.reviewUrl`. 같은 `messageId`는 같은 작업.
- 결과를 사용자에게 줄 때 **화면 링크(`reviewUrl`)를 같이** 준다 — 화면에서 고친 글이 A2A·내려받기에 그대로 반영된다.

## 4. 확인 (CLI는 입력마다 stdout 한 줄 JSON)
- `"ok": true`, 종료 코드 0. 실패면 `error`와 `selection[].why_not`/`run_error`를 그대로 사용자에게 보인다.
- `"warning"`이 있으면 화자 분리가 실패해 화자 없이 받아 적은 것 — 알린다 (무음·음악뿐이면 정상).
- `audio_s`가 원본 길이와 같고 마지막 구간 `end`가 `audio_s` 근처 — 크게 모자라면 뒷부분이 빠진 것 (함정 1). `segments` 0이면 무음·음악뿐인지 알린다.

## 5. 엔진 등록부 (`$ROOT/stt/models.json`) — 모델·경로를 바꾸는 곳
- `engines.<이름>`: `runner`(`qwen`·`faster_whisper`·`diarizen`) · `python`(그 엔진의 환경) 또는 `command` · `model`(HF id 또는 경로) ·
  `languages` · `vram_mib`(필요 GPU — 실측이면 `note`에 날짜·방법). `downloader`: 주소를 받을 yt-dlp.
- `prefer`: 정확도 순서. 조건(언어·모델 있음·환경 있음·GPU 여유)을 통과한 첫 엔진, 실행 중 실패하면 다음 후보 (후보 수가 재시도 상한).
- 경로는 저장소를 만든 PC 기준이다. 다른 PC는 이 파일을 고치거나 `STT_MODELS=<json 경로>`.
- 새 `runner`는 `stt/engine_run.py`의 `load_engine`에 갈래 하나. 고친 뒤 `--dry-run`, `--dry-run --lang xx`(Qwen 미지원 → Whisper)로 선택을 본다.

## 함정
- Qwen3-ASR을 패키지 기본값으로 긴 오디오에 돌리면 뒷부분이 잘린다 (20분 조각 · 출력 512토큰). 5.7분 한국어에서 737자만 나옴 → 15초 조각. 이 값을 크게 올리지 않는다 (F1, D5)
- voice-eval 환경의 faster-whisper에 파일 경로를 넘기면 PyAV 버전이 안 맞아 `metadata_errors` 오류 → numpy로 넘긴다 (F2)
- DiariZen에 `--num-speakers`를 주면 결과가 `<이름>.n2.rttm`으로 바뀐다 → RTTM은 이름 패턴으로 찾는다 (F3)
- "2인 팟캐스트"라 여긴 영상이 실제로 진행자 2 + 대화 예시 2로 4명이었다. 같은 20초 음성도 실행마다 2명·3명으로 갈렸다 → 화자 수를 짐작해 넣지 않고, 점검도 화자 수를 고정해 판정하지 않는다 (F4)
- 쉼(0.6초)에서 잘라 인식하면 단어 오류가 늘었다 (맞음 8→4) → 인식은 15초 조각째로, VAD는 문장 시각 배분에만 (D7)
- 실시간에서 정확히 15초에서 자르면 단어 가운데가 잘렸다 → 끝 2초 안의 가장 조용한 지점에서 끊는다 (F6)
- 말소리가 없는 오디오에서 DiariZen이 `negative dimensions` 오류로 죽는다 → 화자 없이 계속하고 `warning` (F5)
- 같은 GPU를 다른 작업이 쓰면 화자 분리가 13초 → 156초로 느려지고, 여유가 모자라면 "엔진을 쓸 수 없음"으로 막힌다 (설계대로) → 느리면 `nvidia-smi`부터 (C4)

## 관리
실패 신호: 사용자가 받아 적은 글·화자를 많이 고침 / 뒷부분이 빠짐 / 엔진 선택이 의도와 다름 / 다른 PC·모델로 옮길 때 등록부만으로 안 됨.
걸리면 같은 폴더 `NOTES.md`에 [F] 한 줄 (입력 · 증상 · 원인 · 조치 · 상태). 이 스킬을 고치기 전에 `NOTES.md` 색인을 먼저 읽는다.
