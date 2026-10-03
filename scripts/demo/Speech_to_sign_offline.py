import os
import time
import threading
import numpy as np
import cv2
import pyaudio

from SignBridge.whisper_stt import transcribe_frames, load_model, get_engine_info

CHUNK        = 1024
FORMAT       = pyaudio.paInt16
CHANNELS     = 1
RATE         = 16000
GIF_DIR      = "../../assets/signs_gifs"
WORD_MS      = 3000
LETTER_MS    = 2200
DISPLAY_W    = 640
DISPLAY_H    = 480
GAP_MS = 400
recording    = False
speed = 1.0
frames       = []
audio_stream = None
pa           = None
gif_queue    = []
current_text = ""
status_msg   = "Press SPACE to record"

def start_recording():
    global recording, frames, audio_stream, pa, status_msg
    frames = []
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        format=FORMAT, channels=CHANNELS, rate=RATE,
        input=True, frames_per_buffer=CHUNK
    )
    recording = True
    status_msg = "● RECORDING — press SPACE to stop"
    print("[REC] Started")

    def capture():
        while recording:
            try:
                data = audio_stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except Exception as e:
                print(f"[REC] Error: {e}")
                break

    threading.Thread(target=capture, daemon=True).start()


def stop_recording():
    global recording, audio_stream, pa, status_msg, current_text, gif_queue
    recording = False
    time.sleep(0.15)

    if audio_stream:
        audio_stream.stop_stream()
        audio_stream.close()
    if pa:
        pa.terminate()

    if not frames:
        status_msg = "No audio captured"
        return

    duration = len(frames) * CHUNK / RATE
    print(f"[REC] Stopped — {duration:.1f}s captured")
    status_msg = "Transcribing with Whisper..."
    result = transcribe_frames(frames, sample_width=2, channels=CHANNELS, rate=RATE)
    text   = result["text"]

    print(f"[STT] Engine: {result['engine']} | {result['latency_ms']}ms")
    print(f"[STT] Text: {text}")

    if not text:
        status_msg = "Nothing recognised — try again"
        return

    current_text = text
    gif_queue    = build_gif_queue(text)
    status_msg   = f"{result['engine']} · {result['latency_ms']}ms"
    threading.Thread(target=play_queue, daemon=True).start()

def build_gif_queue(text: str) -> list:
    queue = []
    for word in text.strip().split():
        clean = "".join(c for c in word if c.isalnum()).lower()
        if not clean:
            continue

        word_path = os.path.join(GIF_DIR, "words", f"{clean}.gif")
        if os.path.exists(word_path):
            queue.append({"type": "word", "label": word, "path": word_path, "ms": WORD_MS})
        else:
            for ch in clean:
                if ch.isalpha():
                    p = os.path.join(GIF_DIR, "letters", f"{ch.upper()}.gif")
                    if os.path.exists(p):
                        queue.append({"type": "letter", "label": ch.upper(),
                                      "path": p, "ms": LETTER_MS})
    print(f"[GIF] Queue built: {len(queue)} items")
    return queue


def play_queue():
    global gif_queue
    for i, item in enumerate(gif_queue):
        gif_queue[i]["active"] = True
        gif_queue[i]["restart"] = True
        time.sleep((item["ms"] * speed) / 1000)
        gif_queue[i]["active"] = False
        gif_queue[i]["done"]   = True
        time.sleep(GAP_MS / 1000)



def load_gif_frame(path):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None

def decode_gif(path):
    frames_out = []
    cap = cv2.VideoCapture(path)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_out.append(cv2.resize(frame, (DISPLAY_W, DISPLAY_H)))
    cap.release()
    print(f"[GIF] Decoded {os.path.basename(path)} — {len(frames_out)} frames")
    return frames_out

def render():
    global status_msg, speed

    cv2.namedWindow("SignBridge — Speech to Sign (Offline)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SignBridge — Speech to Sign (Offline)", DISPLAY_W, DISPLAY_H + 140)

    gif_frames = {}       # path -> list of decoded frames
    frame_idx  = 0
    last_path  = None
    last_frame = None

    while True:
        canvas = np.full((DISPLAY_H + 140, DISPLAY_W, 3), 18, dtype=np.uint8)
        active = next((it for it in gif_queue if it.get("active")), None)

        if active:
            path = active["path"]

            # Decode this GIF once, cache all frames in memory
            if path not in gif_frames:
                gif_frames[path] = decode_gif(path)

            frames_list = gif_frames[path]

            # New sign started -> reset frame counter
            if path != last_path or active.get("restart"):
                frame_idx = 0
                last_path = path
                active["restart"] = False

            if frames_list:
                frame = frames_list[frame_idx % len(frames_list)]
                frame_idx += 1
                canvas[:DISPLAY_H] = frame
                last_frame = frame

            cv2.putText(canvas, active["label"], (20, DISPLAY_H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (110, 200, 245), 3)

        elif last_frame is not None and gif_queue and \
             not all(it.get("done") for it in gif_queue):
            canvas[:DISPLAY_H] = last_frame

        else:
            cv2.putText(canvas, "Awaiting speech", (DISPLAY_W//2 - 130, DISPLAY_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 90, 90), 2)

        cv2.imshow("SignBridge — Speech to Sign (Offline)", canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            if recording:
                stop_recording()
            else:
                start_recording()
        elif key == ord('c'):
            clear()
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()


def clear():
    global gif_queue, current_text, status_msg
    gif_queue    = []
    current_text = ""
    status_msg   = "Press SPACE to record"


if __name__ == "__main__":
    print("=" * 60)
    print("SignBridge — Speech to Sign (Fully Offline)")
    print("=" * 60)

    print("\nLoading Whisper model...")
    load_model()
    info = get_engine_info()
    print(f"Engine : {info['engine']} ({info['model']})")
    print(f"Offline: {info['offline']}  |  API key required: {info['requires_key']}")

    print("\nControls:")
    print("  SPACE — start / stop recording")
    print("  C     — clear")
    print("  Q     — quit\n")

    render()