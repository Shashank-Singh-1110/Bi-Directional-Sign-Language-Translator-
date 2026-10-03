import os
import time
import tempfile
import wave
import numpy as np

MODEL_SIZE     = os.environ.get("WHISPER_MODEL", "small.en")
SAMPLE_RATE    = 16000
FALLBACK_TO_GOOGLE = True

_model = None
_load_time = None


def load_model(size: str = None):
    global _model, _load_time
    if _model is not None:
        return _model

    size = size or MODEL_SIZE
    t0 = time.time()
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "Whisper not installed. Run:\n"
            "    pip install openai-whisper\n"
            "    brew install ffmpeg"
        )

    print(f"[WHISPER] Loading model '{size}' (first run downloads ~500MB)...")
    _model = whisper.load_model(size)
    _load_time = time.time() - t0
    print(f"[WHISPER] Model loaded in {_load_time:.1f}s")
    return _model


def warm_up():
    load_model()
    silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
    try:
        _model.transcribe(silent, fp16=False, language="en")
        print("[WHISPER] Warm-up complete — ready for requests")
    except Exception as e:
        print(f"[WHISPER] Warm-up skipped: {e}")

def transcribe_array(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    t0 = time.time()

    # Normalize dtype
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if sample_rate != SAMPLE_RATE:
        audio = _resample(audio, sample_rate, SAMPLE_RATE)

    try:
        model = load_model()
        result = model.transcribe(
            audio,
            fp16=False,
            language="en",
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        text = result.get("text", "").strip()
        return {
            "text":       text,
            "engine":     "whisper",
            "latency_ms": int((time.time() - t0) * 1000),
            "model":      MODEL_SIZE,
        }

    except Exception as e:
        print(f"[WHISPER] Error: {e}")
        if FALLBACK_TO_GOOGLE:
            return _google_fallback(audio, t0)
        return {"text": "", "engine": "failed", "latency_ms": 0, "model": MODEL_SIZE}


def transcribe_file(path: str) -> dict:
    t0 = time.time()
    try:
        model = load_model()
        result = model.transcribe(
            path,
            fp16=False,
            language="en",
            condition_on_previous_text=False,
        )
        return {
            "text":       result.get("text", "").strip(),
            "engine":     "whisper",
            "latency_ms": int((time.time() - t0) * 1000),
            "model":      MODEL_SIZE,
        }
    except Exception as e:
        print(f"[WHISPER] File transcription error: {e}")
        return {"text": "", "engine": "failed", "latency_ms": 0, "model": MODEL_SIZE}


def transcribe_frames(frames: list, sample_width: int = 2,
                      channels: int = 1, rate: int = 16000) -> dict:
    raw = b"".join(frames)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # Downmix stereo to mono
    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)

    return transcribe_array(audio, sample_rate=rate)

def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    duration = len(audio) / src_rate
    dst_len  = int(duration * dst_rate)
    src_idx  = np.linspace(0, len(audio) - 1, dst_len)
    return np.interp(src_idx, np.arange(len(audio)), audio).astype(np.float32)


def _google_fallback(audio: np.ndarray, t0: float) -> dict:
    try:
        import speech_recognition as sr
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())

        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)

        os.unlink(tmp_path)
        print("[WHISPER] Fell back to Google STT")
        return {
            "text":       text,
            "engine":     "google",
            "latency_ms": int((time.time() - t0) * 1000),
            "model":      "google-stt",
        }
    except Exception as e:
        print(f"[WHISPER] Google fallback also failed: {e}")
        return {"text": "", "engine": "failed", "latency_ms": 0, "model": "none"}


def get_engine_info() -> dict:
    return {
        "engine":        "whisper",
        "model":         MODEL_SIZE,
        "loaded":        _model is not None,
        "load_time_s":   round(_load_time, 1) if _load_time else None,
        "offline":       True,
        "requires_key":  False,
    }

def benchmark(audio_path: str, sizes: list = None):
    global _model, MODEL_SIZE
    sizes = sizes or ["tiny.en", "base.en", "small.en"]
    results = []

    for size in sizes:
        _model = None
        MODEL_SIZE = size
        t0 = time.time()
        load_model(size)
        load_t = time.time() - t0

        r = transcribe_file(audio_path)
        results.append({
            "model":       size,
            "load_s":      round(load_t, 1),
            "inference_ms": r["latency_ms"],
            "text":        r["text"],
        })
        print(f"  {size:10} | load {load_t:5.1f}s | infer {r['latency_ms']:5}ms | {r['text'][:50]}")

    return results


if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("Whisper Offline STT — SignBridge")
    print("=" * 60)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if "--benchmark" in sys.argv:
            print(f"\nBenchmarking on: {path}\n")
            benchmark(path)
        else:
            r = transcribe_file(path)
            print(f"\nEngine:  {r['engine']}")
            print(f"Model:   {r['model']}")
            print(f"Latency: {r['latency_ms']}ms")
            print(f"Text:    {r['text']}")
    else:
        print("\nLoading model to verify installation...")
        load_model()
        print("\n✓ Whisper is working")
        print(f"\nInfo: {get_engine_info()}")
        print("\nUsage:")
        print("  python whisper_stt.py recording.wav")
        print("  python whisper_stt.py recording.wav --benchmark")