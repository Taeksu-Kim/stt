"""엔진 하나로 받아 적는다 — stt.py 가 엔진의 파이썬 환경으로 부른다. 두 가지 입력:

파일:  <엔진 python> engine_run.py --runner qwen|faster_whisper --model <경로|HF id> --audio a.wav --out r.json
           [--pieces pieces.json] [--lang ko] [--context "제목·용어"] [--chunk-sec 15]
  pieces.json = [{"start","end","speaker"}] (화자 분리 결과). 없으면 오디오 전체를 조용한 지점에서 --chunk-sec 이하로 자른다.
  --chunk-sec 보다 긴 조각도 조용한 지점에서 다시 자른다 — Qwen 을 패키지 기본(20분 조각 · 출력 512토큰)으로 두면
  긴 오디오 뒷부분이 잘린다 (SKILL.md 함정 1).
  결과 JSON: {"language", "segments": [{"start","end","speaker","text","timing"}], "pieces", "load_s", "run_s", "audio_s"}

실시간: <엔진 python> engine_run.py --live --runner ... --model ...   (표준 입력 = PCM16 LE 16kHz mono, 끝까지 흘려 보냄)
  Silero VAD 로 말이 멈추면(0.6초) 끊어 바로 받아 적고, 발화마다 JSON 한 줄을 표준 출력에 낸다:
  {"start","end","text","asr_ms","emitted_at"}. 계속 말하면 --max-utt-sec 에서 끊는다. 입력이 끝나면 남은 말을 마저 내고 끝난다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

SR = 16000
QWEN_LANG = {"zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German", "fr": "French",
             "es": "Spanish", "pt": "Portuguese", "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian",
             "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish", "hi": "Hindi", "ms": "Malay",
             "nl": "Dutch", "sv": "Swedish", "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
             "fil": "Filipino", "fa": "Persian", "el": "Greek", "ro": "Romanian", "hu": "Hungarian", "mk": "Macedonian"}
ISO = {v: k for k, v in QWEN_LANG.items()}
PAD_S = 0.15  # 조각 앞뒤 여유 — 화자 경계에서 첫·끝 음절이 잘리지 않게
SENT_END = re.compile(r"(?<=[.!?。？！])\s+")


# ---------- 엔진 (파일·실시간 공용) ----------
def load_engine(a):
    """(transcribe(wavs) -> [([(조각 안 시작, 끝, 글)], 언어)], 로드 초). qwen 은 조각 하나에 글 하나, whisper 는 문장 구간들."""
    t = time.perf_counter()
    if a.runner == "qwen":
        import torch
        from qwen_asr import Qwen3ASRModel
        m = Qwen3ASRModel.from_pretrained(a.model, device_map="cuda:0", dtype=torch.bfloat16,
                                          max_inference_batch_size=a.batch, max_new_tokens=a.max_new_tokens)
        lang = QWEN_LANG.get(a.lang) if a.lang else None

        def transcribe(wavs):
            outs = m.transcribe([(w, SR) for w in wavs], context=a.context, language=lang) if wavs else []
            return [([(0.0, len(w) / SR, (o.text or "").strip())], (o.language or "").split(",")[0])
                    for w, o in zip(wavs, outs)]
    else:
        try:
            import torch  # noqa: F401 — torch 와 같이 깔린 libcublas 를 ctranslate2 가 찾게 (sonnun stt_server 와 같은 이유)
        except ImportError:
            pass
        from faster_whisper import WhisperModel
        m = WhisperModel(a.model, device="cuda", compute_type="int8_float16")

        def transcribe(wavs):
            res = []
            for w in wavs:  # numpy 로 넘긴다 — 경로를 넘기면 voice-eval 의 PyAV 와 안 맞는다 (SKILL.md 함정 2)
                it, info = m.transcribe(w, language=a.lang, initial_prompt=a.context or None, beam_size=5,
                                        vad_filter=True, condition_on_previous_text=False)
                res.append(([(s.start, s.end, s.text.strip()) for s in it], info.language))
            return res
    return transcribe, time.perf_counter() - t


def top_lang(a, langs: list[str]) -> str:
    langs = [x for x in langs if x]
    top = max(set(langs), key=langs.count) if langs else ""
    return a.lang or ISO.get(top, top)


# ---------- 파일 ----------
def split_long(wav, max_sec: float):
    """(조각 wav, 조각 안 시작 초) 목록. max_sec 이하면 그대로."""
    if len(wav) / SR <= max_sec:
        return [(wav, 0.0)]
    from qwen_asr.inference.utils import split_audio_into_chunks  # 조용한 지점에서 자르는 함수 (두 runner 모두 voice-eval)
    return split_audio_into_chunks(wav, SR, max_chunk_sec=max_sec)


def make_pieces(wav, pieces: list[dict] | None, max_sec: float) -> list[dict]:
    total = len(wav) / SR
    if pieces is None:
        pieces = [{"start": 0.0, "end": total, "speaker": None}] if total > 0 else []
    out = []
    for p in pieces:
        s, e = max(0.0, p["start"] - PAD_S), min(total, p["end"] + PAD_S)
        if e - s < 0.3:
            continue
        for sub, off in split_long(wav[int(s * SR):int(e * SR)], max_sec):
            out.append({"start": s + off, "end": s + off + len(sub) / SR, "speaker": p.get("speaker"), "wav": sub})
    return out


def sentence_times(p: dict, text: str) -> list[dict]:
    """조각 글을 문장으로 나누고 문장 시각을 추정한다 — 글자 비율을 조각 안의 말소리 시간(VAD, 쉼 제외)에 배분.
    인식은 조각째로 하고 VAD 는 시각에만 쓴다: 쉼에서 잘라 인식하면 단어 오류가 늘었다 (NOTES D7)."""
    sents = [s for s in SENT_END.split(text) if s.strip()]
    base = {"speaker": p["speaker"], "start": p["start"], "end": p["end"]}
    if len(sents) < 2:
        return [{**base, "text": text, "timing": "piece"}]
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    spans = [(s["start"] / SR, s["end"] / SR) for s in
             get_speech_timestamps(p["wav"], VadOptions(min_silence_duration_ms=300, speech_pad_ms=100))]
    spans = spans or [(0.0, len(p["wav"]) / SR)]
    total = sum(b - a for a, b in spans)

    def at(frac: float) -> float:  # 말소리 시간의 frac 지점 → 조각 안 시각
        left = frac * total
        for a, b in spans:
            if left <= b - a:
                return a + left
            left -= b - a
        return spans[-1][1]

    n = [len(re.sub(r"\s", "", s)) for s in sents]
    out, done = [], 0
    for s, k in zip(sents, n):
        st, en = at(done / sum(n)), at((done + k) / sum(n))
        done += k
        out.append({**base, "start": p["start"] + st, "end": p["start"] + en, "text": s.strip(), "timing": "estimated"})
    return out


def run_file(a) -> None:
    import soundfile as sf

    wav, sr = sf.read(a.audio, dtype="float32")
    assert sr == SR, f"16kHz 가 아님: {sr}"
    given = json.loads(Path(a.pieces).read_text(encoding="utf-8")) if a.pieces else None
    pieces = make_pieces(wav, given, a.chunk_sec)
    transcribe, load_s = load_engine(a)
    t = time.perf_counter()
    segs, langs = [], []
    for p, (parts, lang) in zip(pieces, transcribe([p["wav"] for p in pieces])):
        langs.append(lang)
        if a.runner == "qwen":
            text = parts[0][2]
            segs += sentence_times(p, text) if text else []
        else:
            segs += [{"start": p["start"] + s, "end": p["start"] + e, "speaker": p["speaker"], "text": x,
                      "timing": "segment"} for s, e, x in parts if x]
    for s in segs:
        s["start"], s["end"] = round(s["start"], 2), round(s["end"], 2)
    Path(a.out).write_text(json.dumps({"language": top_lang(a, langs), "segments": segs, "pieces": len(pieces),
                                       "load_s": round(load_s, 1), "run_s": round(time.perf_counter() - t, 1),
                                       "audio_s": round(len(wav) / SR, 1)}, ensure_ascii=False), encoding="utf-8")


# ---------- 실시간 ----------
class Endpointer:
    """말 시작·끝 판정. 값은 customer-service-agent csagent/voice/vad.py 와 같다 (시작 250ms 이어짐, 끝 600ms 침묵, 앞 여유 300ms).
    Silero 는 faster-whisper 에 든 ONNX 를 쓰고, 청크마다 RNN 상태(h·c)를 이어 간다 (faster-whisper 의 일괄 함수는 상태를 매번 지운다)."""
    CHUNK, CTX = 512, 64

    def __init__(self, max_utt_sec: float, threshold=0.5, neg_threshold=0.35, min_speech_ms=250,
                 min_silence_ms=600, pre_roll_ms=300):
        from faster_whisper.vad import get_vad_model
        self.sess = get_vad_model().session
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.ctx = np.zeros(self.CTX, np.float32)
        per_ms = SR / self.CHUNK / 1000
        self.thr, self.neg = threshold, neg_threshold
        self.min_speech, self.min_silence = int(min_speech_ms * per_ms), int(min_silence_ms * per_ms)
        self.pre_roll, self.max_utt = int(pre_roll_ms * per_ms), int(max_utt_sec * SR / self.CHUNK)
        self.buf = np.zeros(0, np.float32)
        self.n = 0                      # 처리한 청크 수
        self.chunks: list[np.ndarray] = []
        self.probs: list[float] = []
        self.start_n = 0
        self.search = int(2.0 * SR / self.CHUNK)  # 길이로 끊을 때 가장 조용한 지점을 찾는 범위 (끝에서 2초)
        self.run = self.quiet = 0
        self.in_speech = False

    def prob(self, chunk):
        x = np.concatenate([self.ctx, chunk])[None, :]
        out, self.h, self.c = self.sess.run(None, {"input": x, "h": self.h, "c": self.c})
        self.ctx = chunk[-self.CTX:]
        return float(np.asarray(out).reshape(-1)[0])

    def _cut(self, keep_tail: int, carry: int = 0) -> dict:
        """앞에서부터 (전체 - keep_tail - carry) 청크를 발화로 내보낸다. carry 청크는 다음 발화의 시작으로 남긴다."""
        n_out = len(self.chunks) - keep_tail - carry
        audio = np.concatenate(self.chunks[:n_out] or [np.zeros(0, np.float32)])
        start = self.start_n * self.CHUNK / SR
        utt = {"start": round(start, 2), "end": round(start + len(audio) / SR, 2), "wav": audio}
        self.chunks = self.chunks[n_out + keep_tail:] if carry else []
        self.probs = self.probs[n_out + keep_tail:] if carry else []
        self.start_n += n_out + keep_tail
        return utt

    def feed(self, pcm: np.ndarray) -> list[dict]:
        self.buf = np.concatenate([self.buf, pcm])
        out = []
        while len(self.buf) >= self.CHUNK:
            ch, self.buf = self.buf[:self.CHUNK], self.buf[self.CHUNK:]
            p = self.prob(ch)
            self.n += 1
            self.chunks.append(ch)
            self.probs.append(p)
            if not self.in_speech:
                self.run = self.run + 1 if p >= self.thr else 0
                if self.run >= self.min_speech:
                    self.in_speech, self.quiet = True, 0
                    keep = self.min_speech + self.pre_roll
                    self.chunks, self.probs = self.chunks[-keep:], self.probs[-keep:]
                    self.start_n = self.n - len(self.chunks)
                    continue
                keep = self.min_speech + self.pre_roll
                self.chunks, self.probs = self.chunks[-keep:], self.probs[-keep:]
                continue
            self.quiet = self.quiet + 1 if p < self.neg else 0
            if self.quiet >= self.min_silence:            # 말이 멈춤 → 끊는다 (침묵 꼬리는 여유 한 토막만 남김)
                out.append(self._cut(keep_tail=max(0, self.quiet - self.pre_roll)))
                self.in_speech, self.run = False, 0
                self.start_n = self.n
            elif len(self.chunks) >= self.max_utt:         # 쉬지 않고 계속 말함 → 끝 2초 안에서 가장 조용한 지점에서 끊는다
                tail = self.probs[-self.search:]
                quietest = len(self.probs) - len(tail) + int(np.argmin(tail))
                out.append(self._cut(keep_tail=0, carry=len(self.chunks) - quietest - 1))
        return out

    def flush(self) -> list[dict]:
        return [self._cut(keep_tail=0)] if self.in_speech and self.chunks else []


def run_live(a) -> None:
    transcribe, load_s = load_engine(a)
    ep = Endpointer(a.max_utt_sec)
    print(json.dumps({"ready": True, "load_s": round(load_s, 1)}), flush=True)

    def emit(utts):
        for u in utts:
            if len(u["wav"]) < SR * 0.3:
                continue
            t = time.perf_counter()
            parts, lang = transcribe([u["wav"]])[0]
            text = " ".join(x for _, _, x in parts if x).strip()
            if text:
                print(json.dumps({"start": u["start"], "end": u["end"], "text": text, "language": lang,
                                  "asr_ms": round((time.perf_counter() - t) * 1000),
                                  # 입력이 실제 속도로 들어올 때만 뜻이 있다: 첫 소리를 받은 뒤 흐른 시간 - 발화 끝 시각
                                  "lag_s": round(time.perf_counter() - t0[0] - u["end"], 2) if t0 else None,
                                  "emitted_at": time.strftime("%H:%M:%S")}, ensure_ascii=False), flush=True)

    src = sys.stdin.buffer
    t0: list[float] = []
    while True:
        data = src.read(3200)  # 0.1초
        if not data:
            break
        if not t0:
            t0.append(time.perf_counter())
        data = data[:len(data) // 2 * 2]
        emit(ep.feed(np.frombuffer(data, "<i2").astype(np.float32) / 32768.0))
    emit(ep.flush())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True, choices=["qwen", "faster_whisper"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--audio")
    ap.add_argument("--out")
    ap.add_argument("--pieces")
    ap.add_argument("--lang")
    ap.add_argument("--context", default="")
    ap.add_argument("--chunk-sec", type=float, default=15.0)  # 30·15·10초 비교에서 15초가 단어 오류 가장 적음 (NOTES D5)
    ap.add_argument("--max-utt-sec", type=float, default=15.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    a = ap.parse_args()
    if a.live:
        run_live(a)
    else:
        if not (a.audio and a.out):
            ap.error("파일 모드는 --audio 와 --out 이 필요")
        run_file(a)


if __name__ == "__main__":
    main()
