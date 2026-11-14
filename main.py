#!/usr/bin/env python3
"""
ClipForge v2 - find and cut the most engaging clips from long-form video.

Pipeline
  1. (optional) download URL with yt-dlp (+ YouTube "Most replayed" heatmap, live-chat replay)
  2. extract audio -> transcribe with faster-whisper locally OR chunked Cloud Whisper fallback
  3. AUDIENCE ENGAGEMENT curve from:
       - laughter / applause detection in the audio (built-in heuristic, or AST model)
       - YouTube "Most replayed" heatmap (real viewer data)
       - live-chat message density (stream replays)
  4. candidate clips on sentence boundaries, scored on hook, 3-act story arc, pace, energy,
     fillers, crowd reaction + optional user prompt
  5. (optional) Claude re-rank (titles, keywords, virality reasoning, prompt-aware)
  6. top N non-overlapping clips, extended to include the laughter/applause tail
  7. ffmpeg render: 9:16, OpusClip-style animated word-by-word captions (+ .srt sidecar)
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
HOOK_RE = re.compile(
    r"^(here'?s|the (secret|truth|biggest|real|one|worst|best)|nobody|no one|never|stop|why|how|"
    r"what if|did you know|imagine|listen|the problem|most people|everyone|you (need|have|should|won'?t)|"
    r"i (was|got|lost|made|quit|failed|almost)|let me tell)",
    re.I,
)
POWER_WORDS = set(
    """secret truth mistake wrong crazy insane shocking never always biggest worst best money million
    billion free fail failed failure win won lose lost love hate fear afraid angry scared dangerous
    illegal hidden proven stop warning nobody everyone dollars killed died death changed life
    broke rich poor genius stupid terrible amazing incredible unbelievable regret destroyed
    exposed lie lies lied honest honestly literally actually""".split()
)
FILLERS = {"um", "uh", "erm", "like", "basically", "so", "right", "okay", "yeah"}
SENT_END = re.compile(r"[.!?]['\")\]]*\s*$")
CLAUSE_END = re.compile(r"[,;:]['\")\]]*\s*$")

# Caption presets. Colors are BGR hex (ASS format).
STYLES = {
    "pop": dict(upper=True, size=0.052, base="FFFFFF", active="00FFFF", keyword="76E600", outline=0.008,
                box=False, max_words=3, max_chars=18, pop=114, highlight=True, margin=0.24),
    "karaoke": dict(upper=True, size=0.050, base="FFFFFF", active="00C8FF", keyword="00C8FF", outline=0.008,
                    box=False, max_words=4, max_chars=22, pop=100, highlight=True, margin=0.24),
    "boxed": dict(upper=True, size=0.045, base="FFFFFF", active="00FFFF", keyword="76E600", outline=0.014,
                  box=True, max_words=3, max_chars=18, pop=108, highlight=True, margin=0.24),
    "minimal": dict(upper=False, size=0.040, base="FFFFFF", active="FFFFFF", keyword="FFFFFF", outline=0.005,
                    box=False, max_words=6, max_chars=32, pop=100, highlight=False, margin=0.12),
}

CROWD_LABELS = {
    "Laughter", "Applause", "Cheering", "Clapping", "Crowd", "Giggle",
    "Chuckle, chortle", "Belly laugh", "Baby laughter", "Whoop"
}


@dataclass
class Candidate:
    start: float
    end: float
    text: str
    words: list = field(default_factory=list)
    heur: float = 0.0
    crowd: float = 0.0
    reaction: dict = field(default_factory=dict)
    llm: float | None = None
    title: str = ""
    reason: str = ""
    keywords: list = field(default_factory=list)
    tail: float = 0.0
    score: float = 0.0

    @property
    def dur(self):
        return self.end - self.start


# --------------------------------------------------------------------------- #
# Helpers & Cross-Platform Subprocess
# --------------------------------------------------------------------------- #
LOG_CALLBACK = None


def log(msg):
    line = f"[clipforge] {msg}"
    print(line, flush=True)
    callback = LOG_CALLBACK
    if callback:
        try:
            callback(line)
        except Exception:
            pass


def ffmpeg_path():
    """Find FFmpeg across APK native libs, Desktop, Termux, and custom paths."""
    env_bin = os.environ.get("CLIPFORGE_FFMPEG", "").strip()
    if env_bin and Path(env_bin).is_file() and os.access(env_bin, os.X_OK):
        return env_bin
    try:
        from jnius import autoclass
        PythonActivity = autoclass("org.kivy.android.PythonActivity")
        native_dir = PythonActivity.mActivity.getApplicationInfo().nativeLibraryDir
        bundled = Path(native_dir) / "libffmpeg.so"
        if bundled.is_file():
            return str(bundled)
    except Exception:
        pass
    candidate = shutil.which("ffmpeg")
    if candidate:
        return candidate
    for path_str in (
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
        "/data/data/com.termux/files/usr/bin/ffmpeg",
    ):
        if Path(path_str).is_file() and os.access(path_str, os.X_OK):
            return path_str
    raise RuntimeError("FFmpeg was not found.")
def run(cmd, check=True, **kw):
    cmd = list(cmd)
    if cmd and str(cmd[0]).lower() == "ffmpeg":
        cmd[0] = ffmpeg_path()
    elif cmd and str(cmd[0]).lower() == "ffprobe":
        cmd[0] = shutil.which("ffprobe") or "ffprobe"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required tool not found: {cmd[0]}. Install it and try again.") from exc
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}\n{r.stderr[-1800:]}")
    return r


class _YDLLogger:
    def debug(self, msg):
        if msg and not msg.startswith("[debug] "):
            log(msg)

    def warning(self, msg):
        if msg:
            log(f"Warning: {msg}")

    def error(self, msg):
        if msg:
            log(f"yt-dlp: {msg}")


def download(url, workdir: Path) -> Path:
    if not url.startswith(("http://", "https://")):
        raise RuntimeError("Invalid URL. Only http:// and https:// links are supported.")
    log("Downloading video with yt-dlp...")
    out = str(workdir / "source.%(ext)s")
    try:
        import yt_dlp
        has_ffmpeg = shutil.which("ffmpeg") is not None
        ydl_opts = {
            "format": "bv*[height<=1080]+ba/b[height<=1080]/b" if has_ffmpeg else "b/ba/bv*",
            "merge_output_format": "mp4" if has_ffmpeg else None,
            "extractor_args"
            "writeinfojson": True,
            "outtmpl": out,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "noplaylist": True,
            "logger": _YDLLogger(),

            "writeinfojson": True,
            "outtmpl": out,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "noplaylist": True,
            "logger": _YDLLogger(),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        try:
            chat_opts = {
                "skip_download": True,
                "writesubtitles": True,
                "subtitleslangs": ["live_chat"],
                "outtmpl": out,
                "quiet": True,
                "no_warnings": True,
                "noprogress": True,
                "noplaylist": True,
                "logger": _YDLLogger(),
            }
            with yt_dlp.YoutubeDL(chat_opts) as ydl:
                ydl.download([url])
        except Exception:
            pass
    except ImportError:
        run(["yt-dlp", "--no-playlist", "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
             "--merge-output-format", "mp4", "--write-info-json", "-o", out, url])
        run(["yt-dlp", "--no-playlist", "--skip-download", "--write-subs",
             "--sub-langs", "live_chat", "-o", out, url], check=False)

    matches = [p for p in workdir.glob("source.*") if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
    if not matches:
        raise RuntimeError("Download finished, but no video file was found.")
    return max(matches, key=lambda p: p.stat().st_mtime)



def extract_audio(video: Path, wav: Path):
    if wav.exists() and wav.stat().st_size > 1024:
        return
    log("Extracting audio...")
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)])


def read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    return data, sr


def transcribe_cloud_whisper(wav: Path, api_key: str, language: str | None = None):
    """Sample-accurate 9-minute chunked cloud transcription for long videos."""
    is_groq = api_key.startswith("gsk_")
    url = (
        "https://api.groq.com/openai/v1/audio/transcriptions"
        if is_groq else "https://api.openai.com/v1/audio/transcriptions"
    )
    model = "whisper-large-v3-turbo" if is_groq else "whisper-1"
    ffmpeg = ffmpeg_path()
    chunk_seconds = 540
    merged = []

    with tempfile.TemporaryDirectory(prefix="clipforge_chunks_") as td:
        td_path = Path(td)
        with wave.open(str(wav), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            total_frames = wf.getnframes()
            frames_per_chunk = sr * chunk_seconds
            total_chunks = max(1, math.ceil(total_frames / frames_per_chunk))

            for idx in range(total_chunks):
                offset = (idx * frames_per_chunk) / float(sr)
                raw_frames = wf.readframes(frames_per_chunk)
                if not raw_frames:
                    break

                chunk_wav = td_path / f"chunk_{idx:03d}.wav"
                chunk_mp3 = td_path / f"chunk_{idx:03d}.mp3"
                with wave.open(str(chunk_wav), "wb") as out_w:
                    out_w.setnchannels(n_channels)
                    out_w.setsampwidth(sampwidth)
                    out_w.setframerate(sr)
                    out_w.writeframes(raw_frames)

                run([ffmpeg, "-y", "-i", str(chunk_wav), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(chunk_mp3)])
                upload_file = chunk_mp3 if chunk_mp3.exists() else chunk_wav

                log(f"Cloud transcription: audio part {idx + 1}/{total_chunks}...")
                boundary = "----ClipForgeBoundary7MA4YWxkTrZu0gW"
                body = bytearray()

                def add_field(name, value):
                    body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8"))

                add_field("model", model)
                add_field("response_format", "verbose_json")
                add_field("timestamp_granularities[]", "word")
                add_field("timestamp_granularities[]", "segment")
                if language:
                    add_field("language", language)

                body.extend(
                    f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{upload_file.name}"\r\n'
                    f'Content-Type: audio/mpeg\r\n\r\n'.encode("utf-8")
                )
                body.extend(upload_file.read_bytes())
                body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

                req = urllib.request.Request(
                    url,
                    data=bytes(body),
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": f"multipart/form-data; boundary={boundary}",
                        "User-Agent": "ClipForge/2.0",
                    },
                )
                try:
                    with urllib.request.urlopen(req, timeout=240) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"Cloud Whisper API error ({exc.code}): {detail[:1200]}") from exc
                except Exception as exc:
                    raise RuntimeError(f"Cloud Whisper request failed: {exc}") from exc

                segments = payload.get("segments") or []
                if not segments and payload.get("text"):
                    words = payload["text"].split()
                    segments = [{"start": 0.0, "end": max(1.0, len(words) * 0.35), "text": payload["text"]}]

                all_words = payload.get("words") or []
                for seg in segments:
                    s_start = float(seg.get("start", 0.0))
                    s_end = float(seg.get("end", 0.0))
                    start, end = s_start + offset, s_end + offset
                    raw_words = []
                    for word in all_words:
                        ws, we = float(word.get("start", 0.0)), float(word.get("end", 0.0))
                        if ws >= s_start - 0.1 and we <= s_end + 0.1:
                            w_txt = str(word.get("word", "")).strip()
                            if w_txt:
                                raw_words.append({"w": w_txt, "start": ws + offset, "end": we + offset})
                    if not raw_words:
                        tokens = str(seg.get("text", "")).split()
                        step = max((end - start) / max(len(tokens), 1), 0.05)
                        raw_words = [
                            {"w": token, "start": start + j * step, "end": min(end, start + (j + 1) * step)}
                            for j, token in enumerate(tokens)
                        ]
                    merged.append({"start": start, "end": end, "text": str(seg.get("text", "")).strip(), "words": raw_words})
    return merged


def transcribe(wav: Path, cache: Path, model_size: str, language: str | None = None, whisper_api_key: str = ""):
    if cache.exists():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(cached, list) and cached and all(isinstance(x, dict) and "start" in x and "words" in x for x in cached):
                log("Using cached transcript.")
                return cached
        except Exception:
            pass
        log("Transcript cache invalid; rebuilding...")

    try:
        from faster_whisper import WhisperModel
        log(f"Transcribing with Whisper '{model_size}'...")
        model = WhisperModel(model_size, compute_type="auto")
        segs, _ = model.transcribe(str(wav), word_timestamps=True, vad_filter=True, language=language)
        out = [
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip(),
                "words": [{"w": w.word.strip(), "start": float(w.start), "end": float(w.end)} for w in (s.words or [])],
            }
            for s in segs
        ]
        if not out:
            raise RuntimeError("Local Whisper returned no speech segments.")
    except Exception as exc:
        if not whisper_api_key.strip():
            raise RuntimeError(
                f"Local faster-whisper failed ({exc}). Enter a Groq (gsk_...) or OpenAI API key for cloud transcription."
            ) from exc
        log(f"Local Whisper unavailable ({exc}); switching to Cloud Whisper API...")
        out = transcribe_cloud_whisper(wav, whisper_api_key.strip(), language)

    if not out:
        raise RuntimeError("No usable speech detected.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(cache.suffix + ".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    tmp.replace(cache)
    return out


class Energy:
    """Per-0.5s normalized loudness curve (how animated the delivery is)."""

    def __init__(self, data, sr, hop=0.5):
        n = int(sr * hop)
        frames = len(data) // n
        rms = np.sqrt((data[: frames * n].reshape(frames, n) ** 2).mean(axis=1) + 1e-9)
        p95 = np.percentile(rms, 95) or 1.0
        self.curve = np.clip(rms / p95, 0, 1.5)
        self.hop = hop

    def stats(self, start, end):
        a = int(start / self.hop)
        b = max(int(end / self.hop), a + 1)
        seg = self.curve[a:b]
        if len(seg) == 0:
            return 0.0, 0.0
        return float(seg.mean()), float(seg.std())


# --------------------------------------------------------------------------- #
# Audience engagement
# --------------------------------------------------------------------------- #
class Engagement:
    """Combines several 0-1 engagement signals on a shared time grid."""

    def __init__(self, duration, hop=0.5):
        self.hop = hop
        self.n = int(math.ceil(duration / hop)) + 2
        self.sources: dict[str, np.ndarray] = {}

    def add(self, name, arr, normalize=False):
        a = np.zeros(self.n, dtype=np.float32)
        m = min(len(arr), self.n)
        a[:m] = np.asarray(arr[:m], dtype=np.float32)
        if normalize and a.max() > 0:
            a = a / (np.percentile(a, 98) + 1e-9)
        self.sources[name] = np.clip(a, 0, 1)

    @property
    def has_signal(self):
        return any(a.max() > 0.05 for a in self.sources.values())

    @property
    def curve(self):
        if not self.sources:
            return np.zeros(self.n, dtype=np.float32)
        st = np.stack(list(self.sources.values()))
        return 0.6 * st.max(axis=0) + 0.4 * st.mean(axis=0)

    def score(self, start, end):
        c = self.curve
        a = int(max(start, 0) / self.hop)
        b = max(int(end / self.hop), a + 1)
        seg = c[a:b]
        if len(seg) == 0:
            return 0.0
        return float(0.6 * np.percentile(seg, 90) + 0.4 * seg.mean())

    def breakdown(self, start, end):
        a = int(max(start, 0) / self.hop)
        b = max(int(end / self.hop), a + 1)
        return {
            k: int(round(100 * float(np.percentile(v[a:b], 90))))
            for k, v in self.sources.items() if len(v[a:b])
        }

    def tail(self, end, max_ext=3.0, thr=0.35):
        """Seconds to keep after `end` so a laugh/applause reaction isn't cut off."""
        c, ext = self.curve, 0.0
        while ext < max_ext:
            i = int((end + ext) / self.hop)
            if i >= len(c) or c[i] < thr:
                break
            ext += self.hop
        return ext


def crowd_heuristic(data, sr, segs, hop=0.5):
    """Dependency-free laughter + applause detector."""
    B = int(sr * hop)
    nb = len(data) // B
    if nb < 4:
        return {}
    blocks = data[: nb * B].reshape(nb, B)
    rms = np.sqrt((blocks ** 2).mean(axis=1) + 1e-9)
    en = np.clip(rms / (np.percentile(rms, 95) + 1e-9), 0, 1)

    speech = np.zeros(nb, dtype=np.float32)
    for s in segs:
        for w in s["words"]:
            ws, we = w["start"], w["end"]
            for i in range(int(ws / hop), min(int(we / hop) + 1, nb)):
                speech[i] += max(0.0, min(we, (i + 1) * hop) - max(ws, i * hop)) / hop
    speech = np.clip(speech, 0, 1)

    F = 512
    win = np.hanning(F).astype(np.float32)
    lo, hi = int(300 / sr * F), int(6000 / sr * F)
    flat = np.zeros(nb, dtype=np.float32)
    for a in range(0, nb, 800):
        chunk = blocks[a : a + 800, : (B // F) * F].reshape(-1, B // F, F) * win
        p = np.abs(np.fft.rfft(chunk, axis=2)[:, :, lo:hi]) ** 2 + 1e-10
        f = np.exp(np.log(p).mean(axis=2)) / p.mean(axis=2)
        flat[a : a + 800] = f.mean(axis=1)

    E = 640
    ne = len(data) // E
    env = np.sqrt((data[: ne * E].reshape(ne, E) ** 2).mean(axis=1) + 1e-9)
    per = int(hop * sr / E)
    mod = np.zeros(nb, dtype=np.float32)
    half = 25
    for i in range(nb):
        c = int((i + 0.5) * per)
        w = env[max(c - half, 0) : c + half]
        if len(w) < 40:
            continue
        x = w - w.mean()
        den = float((x * x).sum()) + 1e-9
        mod[i] = max(float((x[:-l] * x[l:]).sum()) / den for l in range(4, 9))

    quiet = 1.0 - 0.75 * speech
    applause = np.clip((flat - 0.22) / 0.30, 0, 1) * en * quiet
    laughter = np.clip((mod - 0.20) / 0.45, 0, 1) * en * quiet

    k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    sm = lambda v: np.convolve(v, k, mode="same")
    return {"applause": sm(applause), "laughter": sm(laughter)}


def crowd_ast(data, sr, hop=0.5, win=10.0, stride=5.0):
    """Trained AudioSet classifier (pip install transformers torch)."""
    from transformers import pipeline
    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except ImportError:
        device = -1
    log("Running AST audio classifier for laughter/applause (slow on CPU)...")
    clf = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593", device=device)
    out = np.zeros(int(len(data) / sr / hop) + 2, dtype=np.float32)
    W, S = int(win * sr), int(stride * sr)
    for a in range(0, max(len(data) - W, 1), S):
        chunk = data[a : a + W] / 32768.0
        res = clf({"raw": chunk, "sampling_rate": sr}, top_k=40)
        sc = min(1.0, sum(r["score"] for r in res if r["label"] in CROWD_LABELS) * 1.5)
        i0, i1 = int(a / sr / hop), int((a + W) / sr / hop)
        out[i0:i1] = np.maximum(out[i0:i1], sc)
    return out


def load_heatmap(info_json: Path, hop, n):
    """YouTube 'Most replayed' heatmap from yt-dlp's info json."""
    try:
        info = json.loads(info_json.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None
    hm = info.get("heatmap")
    if not hm:
        return None
    grid = np.zeros(n, dtype=np.float32)
    for seg in hm:
        grid[int(seg["start_time"] / hop) : int(seg["end_time"] / hop) + 1] = seg["value"]
    med = float(np.median(grid[grid > 0])) if (grid > 0).any() else 0.0
    return np.clip((grid - med) / max(1 - med, 1e-3), 0, 1)


def load_chat_density(path: Path, hop, n):
    """Message-per-time density from a yt-dlp live_chat.json replay (JSON lines)."""
    counts = np.zeros(n, dtype=np.float32)
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line).get("replayChatItemAction")
                except Exception:
                    continue
                if not r:
                    continue
                t = int(r.get("videoOffsetTimeMsec", 0)) / 1000
                k = sum(1 for a in r.get("actions", []) if "addChatItemAction" in a)
                if 0 <= t < n * hop:
                    counts[int(t / hop)] += k
    except OSError:
        return None
    if counts.sum() == 0:
        return None
    sm = np.convolve(counts, np.ones(int(10 / hop)) / int(10 / hop), mode="same")
    return sm / (np.percentile(sm, 95) + 1e-9)


# --------------------------------------------------------------------------- #
# Candidate generation + scoring (with Story Arc & Prompt Matching)
# --------------------------------------------------------------------------- #
def build_candidates(segs, min_len, max_len):
    cands = []
    n = len(segs)
    for i in range(n):
        if i > 0 and not SENT_END.search(segs[i - 1]["text"]):
            continue
        for j in range(i, n):
            dur = segs[j]["end"] - segs[i]["start"]
            if dur > max_len:
                break
            if dur >= min_len and SENT_END.search(segs[j]["text"]):
                chunk = segs[i : j + 1]
                words = [w for s in chunk for w in s["words"]]
                if not words:
                    continue
                cands.append(Candidate(
                    start=words[0]["start"], end=words[-1]["end"],
                    text=" ".join(s["text"] for s in chunk), words=words
                ))
    return cands


def prompt_relevance(candidate: Candidate, prompt: str) -> float:
    """Keyword & prefix matching for prompt-based clip selection."""
    terms = [t for t in re.findall(r"[a-z0-9']+", (prompt or "").lower()) if len(t) > 2]
    if not terms:
        return 0.0
    words = re.findall(r"[a-z0-9']+", candidate.text.lower())
    text_spaced = " " + " ".join(words) + " "
    hits = 0.0
    for term in set(terms):
        if f" {term} " in text_spaced:
            hits += 1.0
        elif len(term) >= 4 and any(w.startswith(term[:4]) for w in words):
            hits += 0.6
    return min(1.0, hits / min(len(set(terms)), 4))


def story_arc_score(candidate: Candidate) -> float:
    """3-act heuristic: Setup/Hook -> Conflict/Progression -> Payoff."""
    text = candidate.text.strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) < 2:
        return 0.25
    first = sentences[0].lower()
    middle = " ".join(sentences[1:-1]).lower() if len(sentences) > 2 else text.lower()
    last = sentences[-1].lower()

    hook = bool(HOOK_RE.search(first) or "?" in first or any(w in first for w in ("imagine", "listen", "here's", "secret", "truth")))
    conflict = bool(re.search(r"\b(but|however|suddenly|problem|until|then|actually|turns out|realized|instead)\b", middle))
    payoff = bool(
        SENT_END.search(last)
        and re.search(r"\b(so|because|therefore|that's why|finally|result|lesson|means|realize|turns out)\b", last)
    )
    progression = min(1.0, max(0, len(sentences) - 1) / 3.0)
    return min(1.0, 0.30 * hook + 0.25 * conflict + 0.30 * payoff + 0.15 * progression)


def heuristic_score(c: Candidate, energy: Energy | None, eng: Engagement | None):
    toks = [re.sub(r"[^\w']", "", w["w"].lower()) for w in c.words]
    toks = [t for t in toks if t]
    nwords = max(len(toks), 1)
    dur = max(c.dur, 1)

    first = " ".join(c.text.split()[:18])
    hook = 0.0
    if HOOK_RE.search(first):
        hook += 0.5
    if "?" in first:
        hook += 0.3
    if re.search(r"\d", first) or any(t in POWER_WORDS for t in first.lower().split()):
        hook += 0.2
    hook = min(hook, 1.0)

    power = min(1.0, (sum(t in POWER_WORDS for t in toks) / (nwords / 50)) / 3)
    numbers = min(1.0, len(re.findall(r"\b\d[\d,.]*\b", c.text)) / 3)
    rate = 1 - min(1.0, abs(nwords / dur - 3.0) / 2)
    if energy is not None:
        m, s = energy.stats(c.start, c.end)
        en = min(1.0, m) * 0.6 + min(1.0, s * 2) * 0.4
    else:
        en = 0.5
    complete = 1.0 if SENT_END.search(c.text) else 0.0

    base = 0.30 * hook + 0.15 * power + 0.15 * rate + 0.25 * en + 0.10 * complete + 0.05 * numbers
    if eng is not None and eng.has_signal:
        c.crowd = 100 * eng.score(c.start, c.end + 3.0)
        c.reaction = eng.breakdown(c.start, c.end + 3.0)
        score = 100 * (0.70 * base + 0.30 * c.crowd / 100)
    else:
        score = 100 * base
    score -= 15 * min(1.0, sum(t in FILLERS for t in toks) / nwords * 10)
    arc = story_arc_score(c)
    combined = max(0.0, min(100.0, score)) * 0.88 + arc * 12.0
    return max(0.0, min(100.0, combined))


def overlap(a: Candidate, b: Candidate):
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    return inter / max(min(a.dur, b.dur), 1e-6)


def pick_non_overlapping(cands, k, key, max_overlap=0.2, gap=3.0):
    chosen = []
    for c in sorted(cands, key=key, reverse=True):
        if all(overlap(c, o) <= max_overlap and (c.start >= o.end + gap or c.end <= o.start - gap) for o in chosen):
            chosen.append(c)
        if len(chosen) >= k:
            break
    return chosen


def llm_rerank(cands, model, api_key="", user_prompt=""):
    key = api_key.strip() or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        log("ANTHROPIC_API_KEY not set; skipping LLM rerank.")
        return False

    parts = []
    for i, c in enumerate(cands):
        react = f" | audience reaction (0-100): {c.reaction}" if c.reaction else ""
        parts.append(f"[id={i}] ({c.dur:.0f}s{react})\n{c.text[:1400]}")

    directive = f"\nUSER TOPIC REQUEST: Prioritize clips matching '{user_prompt}'.\n" if user_prompt.strip() else ""
    prompt = f"""You are an expert short-form video editor (TikTok, Reels, Shorts).
Below are candidate clips from a long video. Judge each as a standalone clip on:
- strength of the first 3 seconds (hook / curiosity gap)
- emotional or surprising payoff, story arc, quotable line
- audience reaction when provided (laughter, applause, replay heat = real engagement signals)
- self-contained (makes sense without context), no dead air or rambling
{directive}
Return ONLY a JSON array, one object per clip:
[{{"id": <int>, "score": <0-100>, "title": "<punchy title, max 8 words>", "reason": "<one sentence>",
   "keywords": ["<up to 5 words from the clip worth highlighting in captions>"]}}]

CLIPS:
{chr(10).join(parts)}"""

    log(f"Re-ranking {len(cands)} candidates with {model}...")
    payload = json.dumps({
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            log("Could not parse LLM response; using heuristic scores only.")
            return False
        for r in json.loads(m.group(0)):
            i = r.get("id")
            if isinstance(i, int) and 0 <= i < len(cands):
                cands[i].llm = max(0.0, min(100.0, float(r.get("score", 0))))
                cands[i].title = str(r.get("title", ""))[:120]
                cands[i].reason = str(r.get("reason", ""))[:500]
                cands[i].keywords = [k.lower() for k in r.get("keywords", []) if isinstance(k, str)][:5]
        return True
    except Exception as exc:
        log(f"Claude rerank failed ({exc}); using heuristic scores only.")
        return False


# --------------------------------------------------------------------------- #
# Captions & Rendering
# --------------------------------------------------------------------------- #
def clean_word(w, upper):
    w = w.replace("{", "(").replace("}", ")").replace("\\", "").strip()
    w = re.sub(r"[.,;:]+$", "", w)
    return w.upper() if upper else w


def chunk_words(words, max_words, max_chars, gap=0.45):
    chunks, cur = [], []
    for w in words:
        if cur:
            length = sum(len(x["w"]) for x in cur) + len(cur) + len(w["w"])
            if (len(cur) >= max_words or length > max_chars or w["start"] - cur[-1]["end"] > gap
                    or SENT_END.search(cur[-1]["w"]) or (CLAUSE_END.search(cur[-1]["w"]) and len(cur) >= 2)):
                chunks.append(cur)
                cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def ass_time(t):
    t = max(t, 0)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def srt_time(t):
    t = max(t, 0)
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def is_keyword(word, extra):
    t = re.sub(r"[^\w$%']", "", word.lower())
    return t in POWER_WORDS or t in extra or bool(re.search(r"\d|[$%]", t))


def write_ass(words, clip_start, path: Path, W, H, style, font, extra_kw):
    st = STYLES[style]
    safe_font = re.sub(r"[,;\r\n]+", " ", str(font or "Arial")).strip() or "Arial"
    fs, mv = int(H * st["size"]), int(H * st["margin"])
    outline = max(2, int(H * st["outline"]))
    back = "&H90000000" if st["box"] else "&H80000000"
    out_c = back if st["box"] else "&H00000000"
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 2", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{safe_font},{fs},&H00{st['base']},&H00{st['base']},{out_c},{back},-1,0,0,0,100,100,0,0,"
        f"{3 if st['box'] else 1},{outline},{0 if st['box'] else 2},2,60,60,{mv},1", "",
        "[Events]", "Format: Layer, Start, End, Style, Name, Text",
    ]
    chunks = chunk_words(words, st["max_words"], st["max_chars"])
    for ci, ch in enumerate(chunks):
        next_start = chunks[ci + 1][0]["start"] if ci + 1 < len(chunks) else 1e9
        hold_end = min(ch[-1]["end"] + 0.4, next_start)
        if not st["highlight"]:
            txt = " ".join(clean_word(x["w"], st["upper"]) for x in ch)
            lines.append(f"Dialogue: 0,{ass_time(ch[0]['start'] - clip_start)},{ass_time(hold_end - clip_start)},Default,,{txt}")
            continue
        for k, wd in enumerate(ch):
            t0 = wd["start"] - clip_start
            t1 = (ch[k + 1]["start"] if k + 1 < len(ch) else hold_end) - clip_start
            t1 = max(t1, t0 + 0.06)
            parts = []
            for idx, x in enumerate(ch):
                txt = clean_word(x["w"], st["upper"])
                kw = is_keyword(x["w"], extra_kw)
                if idx == k:
                    col = st["keyword"] if kw else st["active"]
                    p = st["pop"]
                    pop = f"\\fscx100\\fscy100\\t(0,80,\\fscx{p}\\fscy{p})\\t(80,170,\\fscx100\\fscy100)" if p != 100 else ""
                    parts.append(f"{{\\c&H{col}&{pop}}}{txt}{{\\c&H{st['base']}&\\fscx100\\fscy100}}")
                elif kw:
                    parts.append(f"{{\\c&H{st['keyword']}&}}{txt}{{\\c&H{st['base']}&}}")
                else:
                    parts.append(txt)
            lines.append(f"Dialogue: 0,{ass_time(t0)},{ass_time(t1)},Default,,{' '.join(parts)}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_srt(words, clip_start, path: Path):
    out = []
    for i, ch in enumerate(chunk_words(words, 7, 42, gap=0.6), 1):
        out.append(f"{i}\n{srt_time(ch[0]['start'] - clip_start)} --> {srt_time(ch[-1]['end'] - clip_start)}\n{' '.join(x['w'] for x in ch)}\n")
    path.write_text("\n".join(out), encoding="utf-8")


def render_clip(video: Path, c: Candidate, out: Path, fmt: str, style: str | None, font: str):
    start = max(c.start - 0.15, 0)
    dur = max(0.1, c.end + 0.3 + c.tail - start)

    if fmt == "original":
        W, H = 1920, 1080
        graph = "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[v]"
    elif fmt == "crop":
        W, H = 1080, 1920
        graph = "[0:v]crop=trunc(ih*9/16/2)*2:ih,scale=1080:1920[v]"
    else:
        W, H = 1080, 1920
        graph = (
            "[0:v]split[a][b];[a]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,boxblur=25:5[bg];[b]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v]"
        )

    sub = None
    if c.words:
        write_srt(c.words, start, out.with_suffix(".srt"))
        if style:
            sub_path = out.with_suffix(".ass")
            write_ass(c.words, start, sub_path, W, H, style, font, set(c.keywords))
            sub = str(sub_path.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")

    if sub:
        graph += f";[v]subtitles='{sub}'[vsub]"
        out_map = "[vsub]"
    else:
        out_map = "[v]"

    cmd = [
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(video.resolve()), "-t", f"{dur:.3f}",
        "-filter_complex", graph, "-map", out_map, "-map", "0:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-movflags", "+faststart", out.name,
    ]
    run(cmd, cwd=str(out.parent))


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "clip"


def plot_engagement(eng: Engagement, picks, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not installed; skipping plot.")
        return
    c = eng.curve
    t = np.arange(len(c)) * eng.hop / 60
    fig, ax = plt.subplots(figsize=(14, 3.5))
    for name, v in eng.sources.items():
        ax.plot(t, v, lw=0.6, alpha=0.5, label=name)
    ax.plot(t, c, lw=1.2, color="black", label="combined")
    for i, p in enumerate(picks, 1):
        ax.axvspan(p.start / 60, (p.end + p.tail) / 60, color="orange", alpha=0.35)
        ax.text(p.start / 60, 1.02, str(i), fontsize=8)
    ax.set_xlabel("minutes")
    ax.set_ylabel("engagement")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=7, ncol=5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Pipeline Runner
# --------------------------------------------------------------------------- #
def run_pipeline(config: dict):
    inp = str(config.get("input", "")).strip()
    if not inp:
        raise RuntimeError("Enter a YouTube URL or local video path.")
    out = Path(config.get("out") or "clips").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    work.mkdir(exist_ok=True)

    is_url = inp.startswith(("http://", "https://"))
    video = download(inp, work) if is_url else Path(inp).expanduser()
    if not video.is_file():
        raise RuntimeError(f"File not found: {video}")

    wav = work / "audio.wav"
    source_stamp = json.dumps({"input": str(video.resolve()), "size": video.stat().st_size, "mtime": video.stat().st_mtime})
    stamp_file = work / "source_signature.json"
    if stamp_file.exists() and stamp_file.read_text(errors="ignore") != source_stamp:
        for stale in (wav, work / "transcript.json"):
            try:
                stale.unlink()
            except OSError:
                pass
    stamp_file.write_text(source_stamp, encoding="utf-8")

    extract_audio(video, wav)
    segs = transcribe(
        wav,
        work / "transcript.json",
        config.get("whisper_model", "small"),
        config.get("language"),
        config.get("whisper_api_key", ""),
    )
    if not segs:
        raise RuntimeError("No speech detected.")

    data, sr = read_wav(wav)
    energy = Energy(data, sr)
    eng = Engagement(len(data) / sr)

    crowd = config.get("crowd", "heuristic")
    if crowd == "heuristic":
        log("Detecting laughter / applause...")
        for name, arr in crowd_heuristic(data, sr, segs).items():
            eng.add(name, arr)
    elif crowd == "ast":
        try:
            eng.add("crowd_ast", crowd_ast(data, sr))
        except Exception as exc:
            log(f"AST detection skipped ({exc}).")

    info = Path(config["info_json"]) if config.get("info_json") else next(work.glob("source.info.json"), None)
    if info and info.exists():
        hm = load_heatmap(info, eng.hop, eng.n)
        if hm is not None:
            log("Using YouTube 'Most replayed' heatmap.")
            eng.add("replay_heat", hm)

    chat = Path(config["chat"]) if config.get("chat") else next(work.glob("source.live_chat.json"), None)
    if chat and chat.exists():
        cd = load_chat_density(chat, eng.hop, eng.n)
        if cd is not None:
            log("Using live-chat density.")
            eng.add("chat", cd)

    log("Engagement signals: " + (", ".join(eng.sources) or "none") + ("" if eng.has_signal else "  (no crowd reaction detected)"))

    cands = build_candidates(segs, float(config.get("min", 20)), float(config.get("max", 60)))
    if not cands:
        raise RuntimeError("No candidate clips found. Try wider --min/--max.")
    log(f"{len(cands)} candidate windows.")

    prompt = str(config.get("prompt", "")).strip()
    for c in cands:
        c.heur = heuristic_score(c, energy, eng)
        if prompt:
            c.heur = max(0.0, min(100.0, c.heur * 0.75 + prompt_relevance(c, prompt) * 25.0))
        c.score = c.heur

    num = int(config.get("num", 5))
    shortlist = pick_non_overlapping(cands, max(num * 3, 12), key=lambda c: c.heur, max_overlap=0.4)
    if not config.get("no_llm", True) and llm_rerank(
        shortlist,
        config.get("llm_model", "claude-sonnet-4-20250514"),
        config.get("anthropic_key", ""),
        prompt,
    ):
        for c in shortlist:
            if c.llm is None:
                continue
            c.score = (0.30 * c.heur + 0.50 * c.llm + 0.20 * c.crowd) if eng.has_signal else (0.35 * c.heur + 0.65 * c.llm)

    final = pick_non_overlapping(shortlist, num, key=lambda c: c.score)
    final.sort(key=lambda c: c.start)
    if not config.get("no_tail", False) and eng.has_signal:
        for c in final:
            c.tail = eng.tail(c.end)

    style = None if config.get("no_captions", False) else config.get("caption_style", "pop")
    report = []
    for i, c in enumerate(final, 1):
        name = f"{i:02d}_{slug(c.title or c.text[:40])}"
        report.append({
            "file": f"{name}.mp4",
            "start": round(c.start, 2),
            "end": round(c.end + c.tail, 2),
            "duration": round(c.dur + c.tail, 1),
            "score": round(c.score, 1),
            "heuristic": round(c.heur, 1),
            "crowd_engagement": round(c.crowd, 1),
            "reaction": c.reaction,
            "reaction_tail_s": round(c.tail, 1),
            "llm": c.llm,
            "title": c.title,
            "reason": c.reason,
            "keywords": c.keywords,
            "text": c.text,
        })
        if not config.get("analyze_only", False):
            log(f"Rendering {name}.mp4  ({c.dur + c.tail:.0f}s, score {c.score:.0f}, crowd {c.crowd:.0f})")
            render_clip(video, c, out / f"{name}.mp4", config.get("format", "blur"), style, config.get("font", "Arial"))

    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "engagement.json").write_text(json.dumps({
        "hop_seconds": 1.0,
        "sources": list(eng.sources),
        "combined": [round(float(x), 3) for x in eng.curve[:: int(1 / eng.hop)]],
    }), encoding="utf-8")
    if config.get("plot", False):
        plot_engagement(eng, final, out / "engagement.png")
    log(f"Done. Clips + report.json in: {out.resolve()}")
    return report


# --------------------------------------------------------------------------- #
# Android SAF & Runtime Permission Helpers
# --------------------------------------------------------------------------- #
def is_android() -> bool:
    try:
        from kivy.utils import platform
        return platform == "android"
    except Exception:
        return "ANDROID_ARGUMENT" in os.environ or "ANDROID_BOOTLOGO" in os.environ


def request_android_permissions():
    if not is_android():
        return
    try:
        from android.permissions import Permission, request_permissions
        perms = [Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE]
        if hasattr(Permission, "READ_MEDIA_VIDEO"):
            perms.append(Permission.READ_MEDIA_VIDEO)
        request_permissions(perms)
    except Exception as exc:
        log(f"Android permission note: {exc}")


def open_android_saf_video_picker(on_selected_callback, work_cache_dir: Path):
    """Android Storage Access Framework (ACTION_OPEN_DOCUMENT) video picker."""
    if not is_android():
        return False
    try:
        from android import activity
        from jnius import autoclass

        Intent = autoclass("android.content.Intent")
        PythonActivity = autoclass("org.kivy.android.PythonActivity")
        REQ_CODE = 42421

        def _on_activity_result(request_code, result_code, intent):
            if request_code != REQ_CODE:
                return
            try:
                activity.unbind(on_activity_result=_on_activity_result)
            except Exception:
                pass
            if result_code != -1 or intent is None:
                log("Video selection cancelled.")
                return
            try:
                uri = intent.getData()
                if uri is None:
                    return
                context = PythonActivity.mActivity.getApplicationContext()
                resolver = context.getContentResolver()
                in_stream = resolver.openInputStream(uri)
                work_cache_dir.mkdir(parents=True, exist_ok=True)
                dest = work_cache_dir / "saf_selected_video.mp4"
                with open(dest, "wb") as out_f:
                    buf = bytearray(65536)
                    while True:
                        read_len = in_stream.read(buf)
                        if read_len <= 0:
                            break
                        out_f.write(buf[:read_len])
                in_stream.close()
                on_selected_callback(str(dest.resolve()))
                log(f"Imported Android video to cache: {dest.name}")
            except Exception as exc:
                log(f"Android SAF file import failed: {exc}")

        activity.bind(on_activity_result=_on_activity_result)
        intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
        intent.addCategory(Intent.CATEGORY_OPENABLE)
        intent.setType("video/*")
        PythonActivity.mActivity.startActivityForResult(intent, REQ_CODE)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Kivy GUI
# --------------------------------------------------------------------------- #
def launch_gui():
    from kivy.app import App
    from kivy.clock import Clock
    from kivy.core.window import Window
    from kivy.metrics import dp
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.button import Button
    from kivy.uix.checkbox import CheckBox
    from kivy.uix.filechooser import FileChooserListView
    from kivy.uix.gridlayout import GridLayout
    from kivy.uix.label import Label
    from kivy.uix.popup import Popup
    from kivy.uix.scrollview import ScrollView
    from kivy.uix.spinner import Spinner
    from kivy.uix.textinput import TextInput

    Window.clearcolor = (0.09, 0.10, 0.14, 1)

    def default_out_dir():
        android = Path("/storage/emulated/0/Download")
        base = android if android.is_dir() and os.access(android, os.W_OK) else Path.home()
        return str(base / "ClipForge_Clips")

    class ClipForgeApp(App):
        def build(self):
            request_android_permissions()
            self.title = "ClipForge v2 - AI Video Clipper"
            root = BoxLayout(orientation="vertical", padding=dp(10), spacing=dp(7))
            root.add_widget(Label(text="[b]CLIPFORGE v2[/b]  |  AI Video Clipper", markup=True, size_hint_y=None, height=dp(36), color=(0, 0.9, 1, 1)))

            scroll = ScrollView(size_hint=(1, 0.64), bar_width=dp(5))
            form = GridLayout(cols=1, spacing=dp(7), size_hint_y=None, padding=(0, 0, 0, dp(8)))
            form.bind(minimum_height=form.setter("height"))

            def section(text):
                lbl = Label(text=f"[b]{text}[/b]", markup=True, size_hint_y=None, height=dp(27), color=(0.46, 0.9, 0, 1), halign="left", valign="middle")
                lbl.bind(size=lbl.setter("text_size"))
                form.add_widget(lbl)

            def field(label, default="", password=False, hint="", height=dp(58)):
                box = BoxLayout(orientation="vertical", size_hint_y=None, height=height, spacing=dp(2))
                lbl = Label(text=label, size_hint_y=None, height=dp(20), halign="left", valign="middle")
                lbl.bind(size=lbl.setter("text_size"))
                box.add_widget(lbl)
                inp = TextInput(text=str(default), multiline=False, password=password, hint_text=hint, size_hint_y=None, height=dp(36))
                box.add_widget(inp)
                form.add_widget(box)
                return inp

            def browse(target, folder=False):
                if not folder:
                    out_dir = Path(self.output.text.strip() or default_out_dir()).expanduser() / "_work"
                    if open_android_saf_video_picker(lambda p: Clock.schedule_once(lambda *_: setattr(target, "text", p), 0), out_dir):
                        return
                start_dir = "/storage/emulated/0" if Path("/storage/emulated/0").is_dir() else str(Path.home())
                chooser = FileChooserListView(path=start_dir, dirselect=folder)
                content = BoxLayout(orientation="vertical", spacing=dp(5))
                content.add_widget(chooser)
                buttons = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(5))
                popup = Popup(title="Choose folder" if folder else "Choose video file", content=content, size_hint=(0.95, 0.9))

                def select(*_):
                    target.text = chooser.selection[0] if chooser.selection else chooser.path
                    popup.dismiss()

                buttons.add_widget(Button(text="Select", on_release=select))
                buttons.add_widget(Button(text="Cancel", on_release=popup.dismiss))
                content.add_widget(buttons)
                popup.open()

            section("1. Video Source & Output")
            self.source = field("YouTube URL or local video path", hint="Paste YouTube URL or browse a local video")
            row = BoxLayout(size_hint_y=None, height=dp(38), spacing=dp(5))
            row.add_widget(Button(text="Browse video", on_release=lambda *_: browse(self.source, False)))
            form.add_widget(row)

            self.output = field("Output folder", default_out_dir())
            row2 = BoxLayout(size_hint_y=None, height=dp(38), spacing=dp(5))
            row2.add_widget(Button(text="Choose output folder", on_release=lambda *_: browse(self.output, True)))
            form.add_widget(row2)

            section("2. Clip & Visual Settings")
            nums = GridLayout(cols=3, spacing=dp(5), size_hint_y=None, height=dp(58))
            for label, attr, default in (("Clip count", "count", "5"), ("Min seconds", "minimum", "20"), ("Max seconds", "maximum", "60")):
                col = BoxLayout(orientation="vertical")
                col.add_widget(Label(text=label, size_hint_y=None, height=dp(20)))
                w = TextInput(text=default, multiline=False, input_filter="int")
                setattr(self, attr, w)
                col.add_widget(w)
                nums.add_widget(col)
            form.add_widget(nums)

            spins = GridLayout(cols=3, spacing=dp(5), size_hint_y=None, height=dp(58))
            for label, attr, values, default in (
                ("Aspect", "fmt", ["blur", "crop", "original"], "blur"),
                ("Captions", "style", list(STYLES), "pop"),
                ("Crowd", "crowd", ["heuristic", "ast", "off"], "heuristic"),
            ):
                col = BoxLayout(orientation="vertical")
                col.add_widget(Label(text=label, size_hint_y=None, height=dp(20)))
                w = Spinner(text=default, values=values)
                setattr(self, attr, w)
                col.add_widget(w)
                spins.add_widget(col)
            form.add_widget(spins)

            self.prompt = field("Prompt-based clipping (optional)", hint="e.g. funny moments, hot takes, surprising facts")

            section("3. Optional AI & Transcription")
            self.whisper_key = field("Groq / OpenAI API key", os.environ.get("GROQ_API_KEY", ""), True, "Needed on Android if local Whisper fails")
            self.claude_key = field("Anthropic API key (optional)", os.environ.get("ANTHROPIC_API_KEY", ""), True, "Optional Claude reranking")

            self.toggles = {}
            toggles = GridLayout(cols=2, spacing=dp(4), size_hint_y=None, height=dp(90))
            for key, label, default in (
                ("captions", "Burn animated captions", True),
                ("tail", "Keep reaction tail", True),
                ("plot", "Save engagement plot", True),
                ("analyze", "Analyze only (no MP4 render)", False),
            ):
                t_row = BoxLayout(spacing=dp(4))
                cb = CheckBox(active=default, size_hint_x=None, width=dp(34))
                lbl = Label(text=label, halign="left", valign="middle")
                lbl.bind(size=lbl.setter("text_size"))
                t_row.add_widget(cb)
                t_row.add_widget(lbl)
                toggles.add_widget(t_row)
                self.toggles[key] = cb
            form.add_widget(toggles)

            scroll.add_widget(form)
            root.add_widget(scroll)

            self.run_button = Button(text="FORGE CLIPS", size_hint_y=None, height=dp(46), bold=True, background_normal="", background_color=(0.0, 0.68, 0.85, 1))
            self.run_button.bind(on_release=self.start)
            root.add_widget(self.run_button)

            self.console = TextInput(text="[clipforge] Ready. Add a YouTube link or local video file.\n", readonly=True, size_hint=(1, 0.28), background_color=(0.04, 0.05, 0.07, 1), foreground_color=(0.4, 1, 0.5, 1))
            root.add_widget(self.console)

            global LOG_CALLBACK
            LOG_CALLBACK = self.append_log
            return root

        def append_log(self, message):
            Clock.schedule_once(lambda *_: setattr(self.console, "text", self.console.text + message + "\n"), 0)

        def start(self, *_):
            try:
                count = int(self.count.text or 0)
                minimum = float(self.minimum.text or 0)
                maximum = float(self.maximum.text or 0)
                if not self.source.text.strip():
                    raise ValueError("Enter a YouTube URL or select a local video.")
                if count < 1 or count > 100:
                    raise ValueError("Clip count must be between 1 and 100.")
                if minimum <= 0 or maximum <= minimum:
                    raise ValueError("Durations must be positive and Max must be greater than Min.")
                output = self.output.text.strip() or default_out_dir()
                Path(output).expanduser().mkdir(parents=True, exist_ok=True)
                config = {
                    "input": self.source.text.strip(),
                    "out": output,
                    "num": count,
                    "min": minimum,
                    "max": maximum,
                    "format": self.fmt.text,
                    "caption_style": self.style.text,
                    "font": "Arial",
                    "no_captions": not self.toggles["captions"].active,
                    "crowd": self.crowd.text,
                    "no_tail": not self.toggles["tail"].active,
                    "plot": self.toggles["plot"].active,
                    "whisper_model": "small",
                    "language": None,
                    "no_llm": not bool(self.claude_key.text.strip()),
                    "llm_model": os.environ.get("CLIPFORGE_LLM", "claude-sonnet-4-20250514"),
                    "analyze_only": self.toggles["analyze"].active,
                    "whisper_api_key": self.whisper_key.text.strip(),
                    "anthropic_key": self.claude_key.text.strip(),
                    "prompt": self.prompt.text.strip(),
                }
            except Exception as exc:
                self.append_log(f"[ERROR] {exc}")
                return

            self.run_button.disabled = True
            self.run_button.text = "PROCESSING..."

            def worker():
                try:
                    run_pipeline(config)
                except Exception as exc:
                    self.append_log(f"[ERROR] {exc}")
                finally:
                    def restore(_):
                        self.run_button.disabled = False
                        self.run_button.text = "FORGE CLIPS"
                    Clock.schedule_once(restore, 0)

            threading.Thread(target=worker, daemon=True).start()

    ClipForgeApp().run()


# --------------------------------------------------------------------------- #
# Main CLI Entrypoint
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Cut the most engaging clips from long videos.")
    p.add_argument("input", nargs="?", help="video file path or URL (omit to launch GUI)")
    p.add_argument("-o", "--out", default="clips", help="output folder")
    p.add_argument("-n", "--num", type=int, default=5, help="number of clips")
    p.add_argument("--min", type=float, default=20, help="min clip seconds")
    p.add_argument("--max", type=float, default=60, help="max clip seconds")
    p.add_argument("--format", choices=["blur", "crop", "original"], default="blur")
    p.add_argument("--caption-style", choices=list(STYLES), default="pop")
    p.add_argument("--font", default="Arial")
    p.add_argument("--no-captions", action="store_true")
    p.add_argument("--crowd", choices=["heuristic", "ast", "off"], default="heuristic")
    p.add_argument("--info-json")
    p.add_argument("--chat")
    p.add_argument("--no-tail", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--whisper-model", default="small")
    p.add_argument("--language", default=None)
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--llm-model", default=os.environ.get("CLIPFORGE_LLM", "claude-sonnet-4-20250514"))
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--prompt", default="", help="topic/style prompt, e.g. 'funny moments'")
    a = p.parse_args()

    if a.input:
        config = vars(a)
        config["whisper_api_key"] = os.environ.get("GROQ_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
        config["anthropic_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
        run_pipeline(config)
    else:
        launch_gui()


if __name__ == "__main__":
    main()