import cv2
import numpy as np
import speech_recognition as sr
import threading
import time
import os
import urllib.request
import urllib.error
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

GIF_DIR      = 'signs_gifs'
WINDOW_NAME  = 'Speech to Sign Language'
DISPLAY_SIZE = (640, 480)
FRAME_DELAY  = 0.08
WORD_PAUSE   = 0.3
LETTER_PAUSE = 0.15

LETTER_URLS = {
    'A': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/a.gif',
    'B': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/b.gif',
    'C': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/c.gif',
    'D': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/d.gif',
    'E': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/e.gif',
    'F': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/f.gif',
    'G': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/g.gif',
    'H': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/h.gif',
    'I': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/i.gif',
    'J': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/j.gif',
    'K': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/k.gif',
    'L': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/l.gif',
    'M': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/m.gif',
    'N': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/n.gif',
    'O': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/o.gif',
    'P': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/p.gif',
    'Q': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/q.gif',
    'R': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/r.gif',
    'S': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/s.gif',
    'T': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/t.gif',
    'U': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/u.gif',
    'V': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/v.gif',
    'W': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/w.gif',
    'X': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/x.gif',
    'Y': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/y.gif',
    'Z': 'https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/z.gif',
}

NUMBER_URLS = {
    '0': 'https://lifeprint.com/asl101/signjpegs/numbers/number00.jpg',
    '1': 'https://lifeprint.com/asl101/signjpegs/numbers/number01.jpg',
    '2': 'https://lifeprint.com/asl101/signjpegs/numbers/number02.jpg',
    '3': 'https://lifeprint.com/asl101/signjpegs/numbers/number03.jpg',
    '4': 'https://lifeprint.com/asl101/signjpegs/numbers/number04.jpg',
    '5': 'https://lifeprint.com/asl101/signjpegs/numbers/number05.jpg',
    '6': 'https://lifeprint.com/asl101/signjpegs/numbers/number06.jpg',
    '7': 'https://lifeprint.com/asl101/signjpegs/numbers/number07.jpg',
    '8': 'https://lifeprint.com/asl101/signjpegs/numbers/number08.jpg',
    '9': 'https://lifeprint.com/asl101/signjpegs/numbers/number09.jpg',
}


def download_gifs():
    os.makedirs(os.path.join(GIF_DIR, 'letters'), exist_ok=True)
    os.makedirs(os.path.join(GIF_DIR, 'numbers'), exist_ok=True)
    os.makedirs(os.path.join(GIF_DIR, 'words'),   exist_ok=True)

    all_downloads = []
    for letter, url in LETTER_URLS.items():
        all_downloads.append((letter, url, os.path.join(GIF_DIR, 'letters', f'{letter}.gif')))
    for num, url in NUMBER_URLS.items():
        all_downloads.append((num, url, os.path.join(GIF_DIR, 'numbers', f'{num}.gif')))

    print(f"\n[INFO] Checking {len(all_downloads)} GIFs...")
    success, failed = 0, []
    for name, url, path in all_downloads:
        if os.path.exists(path):
            success += 1
            continue
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as r:
                with open(path, 'wb') as f:
                    f.write(r.read())
            success += 1
            print(f"  ✓ {name}")
        except Exception as e:
            failed.append(name)
            print(f"  ✗ {name} — {e}")
    print(f"[INFO] Ready: {success}/{len(all_downloads)}")
    if failed:
        print(f"[WARN] Failed: {failed}")

def load_gif_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, DISPLAY_SIZE))
    cap.release()

    if not frames:
        try:
            from PIL import Image
            img = Image.open(path)
            for i in range(getattr(img, 'n_frames', 1)):
                img.seek(i)
                frame = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
                frames.append(cv2.resize(frame, DISPLAY_SIZE))
        except Exception:
            pass
    return frames


def get_gif_path(token):
    token = token.strip().lower()
    word_path = os.path.join(GIF_DIR, 'words', f'{token}.gif')
    if os.path.exists(word_path):
        return [word_path]
    if len(token) == 1 and token.isalpha():
        p = os.path.join(GIF_DIR, 'letters', f'{token.upper()}.gif')
        return [p] if os.path.exists(p) else None
    if len(token) == 1 and token.isdigit():
        p = os.path.join(GIF_DIR, 'numbers', f'{token}.gif')
        return [p] if os.path.exists(p) else None
    paths = []
    for ch in token:
        if ch.isalpha():
            p = os.path.join(GIF_DIR, 'letters', f'{ch.upper()}.gif')
            if os.path.exists(p):
                paths.append(p)
        elif ch.isdigit():
            p = os.path.join(GIF_DIR, 'numbers', f'{ch}.gif')
            if os.path.exists(p):
                paths.append(p)
    return paths if paths else None


def text_to_sign_sequence(text):
    words = text.strip().split()
    gif_list, labels = [], []
    for word in words:
        paths = get_gif_path(word)
        if paths:
            if len(paths) == 1:
                gif_list.append(paths[0])
                labels.append(word)
            else:
                for i, path in enumerate(paths):
                    gif_list.append(path)
                    labels.append(f"{word}[{word[i].upper()}]")
        else:
            for ch in word:
                if ch.isalpha():
                    p = os.path.join(GIF_DIR, 'letters', f'{ch.upper()}.gif')
                    if os.path.exists(p):
                        gif_list.append(p)
                        labels.append(ch.upper())
    return gif_list, labels

class PushToTalkRecorder:
    def __init__(self):
        self.is_recording = False
        self._frames     = []
        self._thread     = None
        self._stop_event = threading.Event()
        self._rate       = 16000
        self._chunk      = 1024

    def start(self):
        import pyaudio
        self._frames = []
        self._stop_event.clear()
        self.is_recording = True

        def _record():
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self._rate,
                input=True,
                frames_per_buffer=self._chunk
            )
            print("[REC] ● Recording...")
            while not self._stop_event.is_set():
                data = stream.read(self._chunk, exception_on_overflow=False)
                self._frames.append(data)
            stream.stop_stream()
            stream.close()
            pa.terminate()
            print("[REC] ■ Stopped")

        self._thread = threading.Thread(target=_record, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self.is_recording = False
        if self._thread:
            self._thread.join(timeout=2)

        if not self._frames:
            return ""

        import pyaudio, io, wave
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # paInt16 = 2 bytes
            wf.setframerate(self._rate)
            wf.writeframes(b''.join(self._frames))
        buf.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(buf) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"[STT] Recognized: '{text}'")
            return text
        except sr.UnknownValueError:
            print("[STT] Could not understand")
            return ""
        except sr.RequestError as e:
            print(f"[STT] Error: {e}")
            return ""

class SignDisplay:
    def __init__(self):
        self.current_text  = ""
        self.status        = "SPACE: start speaking"
        self.is_playing    = False
        self.stop_playback = False
        self.frame_to_show = self._blank_frame()
        self.lock          = threading.Lock()

    def _blank_frame(self):
        frame = np.zeros((*DISPLAY_SIZE[::-1], 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)
        return frame

    def _draw_ui(self, frame, word_label=""):
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, self.status, (15, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 100), 2)

        if word_label:
            cv2.putText(frame, word_label.upper(), (15, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        if self.current_text:
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (0, h - 40), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, self.current_text, (15, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)

        cv2.putText(frame, "SPACE: start/stop recording   C: clear   Q: quit",
                    (15, h - 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (80, 80, 80), 1)
        return frame

    def play_sequence(self, gif_paths, labels):
        def _play():
            self.is_playing    = True
            self.stop_playback = False
            for path, label in zip(gif_paths, labels):
                if self.stop_playback:
                    break
                frames = load_gif_frames(path)
                if not frames:
                    continue
                for _ in range(2):
                    for f in frames:
                        if self.stop_playback:
                            break
                        display = self._draw_ui(f.copy(), label)
                        with self.lock:
                            self.frame_to_show = display
                        time.sleep(FRAME_DELAY)
                time.sleep(LETTER_PAUSE)
            self.is_playing = False
            self.status = "SPACE: start speaking"
            with self.lock:
                self.frame_to_show = self._draw_ui(self._blank_frame())

        threading.Thread(target=_play, daemon=True).start()

    def get_frame(self):
        with self.lock:
            return self.frame_to_show.copy()


def main():
    print("=" * 55)
    print("  SPEECH TO SIGN LANGUAGE  —  Push-to-Talk")
    print("  SPACE: start/stop recording")
    print("  C: clear   Q: quit")
    print("=" * 55)

    download_gifs()

    display  = SignDisplay()
    recorder = PushToTalkRecorder()

    with display.lock:
        display.frame_to_show = display._draw_ui(display._blank_frame())

    cv2.namedWindow(WINDOW_NAME)
    print("\n[READY] Press SPACE in the window to start speaking")

    while True:
        frame = display.get_frame()
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'):
            recorder._stop_event.set()
            recorder.is_recording = False
            display.stop_playback  = True
            display.current_text   = ""
            display.status         = "SPACE: start speaking"
            with display.lock:
                display.frame_to_show = display._draw_ui(display._blank_frame())
            print("[INFO] Cleared")

        elif key == ord(' '):
            if recorder.is_recording:
                display.status = "Processing..."
                with display.lock:
                    display.frame_to_show = display._draw_ui(display._blank_frame())

                def process():
                    text = recorder.stop()
                    if text:
                        display.current_text = text
                        display.status = f'Signing: "{text}"'
                        gif_list, labels = text_to_sign_sequence(text)
                        if gif_list:
                            display.play_sequence(gif_list, labels)
                        else:
                            display.status = "No signs found — try again"
                            with display.lock:
                                display.frame_to_show = display._draw_ui(display._blank_frame())
                    else:
                        display.status = "Not understood — try again"
                        with display.lock:
                            display.frame_to_show = display._draw_ui(display._blank_frame())

                threading.Thread(target=process, daemon=True).start()

            else:
                if display.is_playing:
                    display.stop_playback = True
                    time.sleep(0.1)
                display.current_text = ""
                display.status = "● RECORDING — press SPACE to stop"
                with display.lock:
                    display.frame_to_show = display._draw_ui(display._blank_frame())
                recorder.start()

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == '__main__':
    main()