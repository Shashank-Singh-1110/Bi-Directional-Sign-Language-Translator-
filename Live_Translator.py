import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import os
import socket
import json
import collections
import queue
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.keras.models import load_model
import pyttsx3
import speech_recognition as sr

MODEL_PATH      = 'action_norm.h5'
GIF_DIR         = 'signs_gifs'
PORT            = 9999
SEQUENCE_LENGTH = 30
STABILITY_FRAMES= 10
COOLDOWN_SECONDS= 3.0
FEATURE_DIM     = 126
FRAME_DELAY     = 0.08
LETTER_PAUSE    = 0.3
DISPLAY_W       = 1280
DISPLAY_H       = 480

GESTURE_SIGNS   = {'Hello', 'Thanks', 'Yes', 'I LOVE YOU', 'No', 'Sorry'}
GESTURE_THRESH  = 0.75
LETTER_THRESH   = 0.92
MIN_CONF_GAP    = 0.25
MOTION_FRAMES   = 6
MOTION_THRESH   = 0.012

ACTIONS = np.array([
    'Hello', 'Thanks', 'Yes', 'I LOVE YOU', 'No', 'Sorry',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
])

mp_holistic   = mp.solutions.holistic
mp_drawing    = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.my_current_word   = ""
        self.my_confidence     = 0.0
        self.my_sentence       = []
        self.my_hand_still     = False
        self.my_cam_frame      = None
        self.peer_sentence     = ""
        self.peer_gif_label    = ""
        self.peer_gif_frame    = None
        self.peer_is_playing   = False
        self.peer_stop_gif     = False
        self.connected         = False
        self.peer_ip           = ""
        self.sign_to_send_q    = queue.Queue()
        self.speech_to_send_q  = queue.Queue()
        self.received_sign_q   = queue.Queue()
        self.received_speech_q = queue.Queue()

state = SharedState()

def normalize_hand(hand_63):
    pts   = hand_63.reshape(21, 3)
    wrist = pts[0].copy()
    pts   = pts - wrist
    max_d = np.linalg.norm(pts, axis=1).max()
    if max_d > 1e-6:
        pts = pts / max_d
    return pts.flatten()

def extract_keypoints(results):
    lh = np.array([[lm.x, lm.y, lm.z]
                   for lm in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[lm.x, lm.y, lm.z]
                   for lm in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([normalize_hand(lh), normalize_hand(rh)])

def is_hand_still(sequence):
    if len(sequence) < MOTION_FRAMES:
        return False
    arr = np.array(list(sequence)[-MOTION_FRAMES:])
    return np.std(arr, axis=0).mean() < MOTION_THRESH

def load_gif_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (DISPLAY_W // 2, DISPLAY_H)))
    cap.release()
    if not frames:
        try:
            from PIL import Image
            img = Image.open(path)
            for i in range(getattr(img, 'n_frames', 1)):
                img.seek(i)
                frame = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
                frames.append(cv2.resize(frame, (DISPLAY_W // 2, DISPLAY_H)))
        except Exception:
            pass
    return frames

def get_gif_paths(text):
    words = text.strip().split()
    result = []
    for word in words:
        word_lower = word.lower()
        word_path  = os.path.join(GIF_DIR, 'words', f'{word_lower}.gif')
        if os.path.exists(word_path):
            result.append((word_path, word))
            continue
        for ch in word:
            if ch.isalpha():
                p = os.path.join(GIF_DIR, 'letters', f'{ch.upper()}.gif')
                if os.path.exists(p):
                    result.append((p, ch.upper()))
    return result

def speak(text):
    def _run():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_run, daemon=True).start()


def send_message(sock, msg_type, payload):
    try:
        msg = json.dumps({"type": msg_type, "data": payload}) + "\n"
        sock.sendall(msg.encode())
    except Exception as e:
        print(f"  [NET] Send error: {e}")

def receive_messages(sock):
    buf = ""
    while True:
        try:
            data = sock.recv(4096).decode()
            if not data:
                break
            buf += data
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                if not line.strip():
                    continue
                msg = json.loads(line)
                mtype = msg["type"]
                mdata = msg["data"]

                if mtype == "sign_sentence":
                    state.received_sign_q.put(mdata)
                    print(f"  [RECV] Sign sentence: '{mdata}'")

                elif mtype == "speech_text":
                    state.received_speech_q.put(mdata)
                    print(f"  [RECV] Speech text: '{mdata}'")

        except Exception as e:
            print(f"  [NET] Recv error: {e}")
            break

    print("  [NET] Connection closed")
    with state.lock:
        state.connected = False

def start_server(peer_sock_holder):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('', PORT))
    server.listen(1)
    my_ip = socket.gethostbyname(socket.gethostname())
    print(f"\n  [NET] Waiting for peer on {my_ip}:{PORT} ...")
    conn, addr = server.accept()
    print(f"  [NET] Connected to {addr[0]}")
    peer_sock_holder.append(conn)
    with state.lock:
        state.connected = True
        state.peer_ip   = addr[0]

def start_client(peer_ip, peer_sock_holder):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"\n  [NET] Connecting to {peer_ip}:{PORT} ...")
    while True:
        try:
            sock.connect((peer_ip, PORT))
            print(f"  [NET] Connected to {peer_ip}")
            peer_sock_holder.append(sock)
            with state.lock:
                state.connected = True
                state.peer_ip   = peer_ip
            break
        except Exception:
            print("  [NET] Retrying...")
            time.sleep(1)



def sign_detection_thread(model, peer_sock_holder):
    cap = cv2.VideoCapture(0)
    sequence         = collections.deque(maxlen=SEQUENCE_LENGTH)
    stability_buffer = collections.deque(maxlen=STABILITY_FRAMES)
    last_pred_time   = 0
    top_preds        = []

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            # Draw landmarks
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_draw_style.get_default_hand_landmarks_style(),
                    mp_draw_style.get_default_hand_connections_style())
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_draw_style.get_default_hand_landmarks_style(),
                    mp_draw_style.get_default_hand_connections_style())

            keypoints  = extract_keypoints(results)
            sequence.append(keypoints)
            hand_still = is_hand_still(sequence)

            if len(sequence) == SEQUENCE_LENGTH:
                X       = np.expand_dims(np.array(sequence), axis=0)
                pred    = model.predict(X, verbose=0)[0]
                top3    = np.argsort(pred)[-3:][::-1]
                top_preds = [(ACTIONS[i], pred[i]) for i in top3]

                best_word = top_preds[0][0]
                best_conf = float(top_preds[0][1])
                sec_conf  = float(top_preds[1][1])
                gap_ok    = (best_conf - sec_conf) >= MIN_CONF_GAP

                hand_visible = (results.left_hand_landmarks is not None or
                                results.right_hand_landmarks is not None)
                threshold    = GESTURE_THRESH if best_word in GESTURE_SIGNS \
                               else LETTER_THRESH

                with state.lock:
                    state.my_current_word = best_word
                    state.my_confidence   = best_conf
                    state.my_hand_still   = hand_still

                if hand_visible and best_conf >= threshold and gap_ok and hand_still:
                    stability_buffer.append(best_word)
                else:
                    stability_buffer.append("")

                stable = (
                    len(stability_buffer) == STABILITY_FRAMES and
                    len(set(stability_buffer)) == 1 and
                    stability_buffer[0] != ""
                )

                now = time.time()
                if stable and (now - last_pred_time) > COOLDOWN_SECONDS:
                    word = stability_buffer[0]
                    with state.lock:
                        if not state.my_sentence or state.my_sentence[-1] != word:
                            state.my_sentence.append(word)
                            if len(state.my_sentence) > 10:
                                state.my_sentence = state.my_sentence[-10:]
                        sentence_str = ' '.join(state.my_sentence)

                    if peer_sock_holder and state.connected:
                        send_message(peer_sock_holder[0], "sign_sentence", sentence_str)
                        print(f"  [SIGN] Sent: '{sentence_str}'")

                    speak(word)
                    last_pred_time = now
                    stability_buffer.clear()
            cam_display = cv2.resize(frame, (DISPLAY_W // 2, DISPLAY_H))
            h, w = cam_display.shape[:2]
            with state.lock:
                word = state.my_current_word
                conf = state.my_confidence
                sent = state.my_sentence.copy()
                still = state.my_hand_still

            overlay = cam_display.copy()
            cv2.rectangle(overlay, (0,0), (w, 80), (20,20,20), -1)
            cv2.addWeighted(overlay, 0.6, cam_display, 0.4, 0, cam_display)
            cv2.putText(cam_display, "SIGNING (You)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
            if word:
                col = (0,200,100) if conf >= (GESTURE_THRESH if word in GESTURE_SIGNS else LETTER_THRESH) else (0,165,255)
                cv2.putText(cam_display, f"{word}  {conf*100:.0f}%", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)

            sc = (0,255,0) if still else (0,0,255)
            cv2.putText(cam_display, "STILL" if still else "MOVING",
                (w-90, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sc, 2)
            overlay2 = cam_display.copy()
            cv2.rectangle(overlay2, (0, h-40), (w, h), (20,20,20), -1)
            cv2.addWeighted(overlay2, 0.7, cam_display, 0.3, 0, cam_display)
            sent_str = ' '.join(sent) if sent else "Sign something..."
            cv2.putText(cam_display, sent_str[:45], (10, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            with state.lock:
                state.my_cam_frame = cam_display

    cap.release()


def gif_display_thread():
    blank = np.zeros((DISPLAY_H, DISPLAY_W // 2, 3), dtype=np.uint8)
    blank[:] = (30, 30, 30)
    peer_frame = blank.copy()
    h, w = peer_frame.shape[:2]
    cv2.putText(peer_frame, "PEER SIGNS", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
    cv2.putText(peer_frame, "Waiting for connection...", (10, h//2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)
    with state.lock:
        state.peer_gif_frame = peer_frame

    while True:
        try:
            sentence = state.received_sign_q.get(timeout=0.1)
        except queue.Empty:
            continue

        gif_seq = get_gif_paths(sentence)
        if not gif_seq:
            continue

        with state.lock:
            state.peer_is_playing = True
            state.peer_stop_gif   = False
            state.peer_sentence   = sentence

        for path, label in gif_seq:
            if state.peer_stop_gif:
                break
            frames = load_gif_frames(path)
            if not frames:
                continue
            with state.lock:
                state.peer_gif_label = label
            for _ in range(2):
                for f in frames:
                    if state.peer_stop_gif:
                        break
                    display = f.copy()
                    h2, w2  = display.shape[:2]
                    overlay = display.copy()
                    cv2.rectangle(overlay, (0,0), (w2,80), (20,20,20), -1)
                    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
                    cv2.putText(display, "PEER SIGNS (Received)", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,220,255), 2)
                    cv2.putText(display, label, (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
                    overlay2 = display.copy()
                    cv2.rectangle(overlay2, (0,h2-40),(w2,h2),(20,20,20),-1)
                    cv2.addWeighted(overlay2,0.7,display,0.3,0,display)
                    cv2.putText(display, sentence[:45], (10,h2-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                    with state.lock:
                        state.peer_gif_frame = display
                    time.sleep(FRAME_DELAY)
            time.sleep(LETTER_PAUSE)

        with state.lock:
            state.peer_is_playing = False
            reset = blank.copy()
            cv2.putText(reset, "PEER SIGNS", (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,220,255), 2)
            cv2.putText(reset, f"Last: '{sentence[:35]}'",
                (10, DISPLAY_H//2), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (100,100,100), 1)
            state.peer_gif_frame = reset

def tts_receiver_thread():
    while True:
        try:
            text = state.received_speech_q.get(timeout=0.1)
        except queue.Empty:
            continue

        print(f"  [TTS] Peer said: '{text}'")
        speak(text)

        gif_seq = get_gif_paths(text)
        if gif_seq:
            state.received_sign_q.put(text)

class PushToTalkRecorder:
    def __init__(self):
        self.is_recording  = False
        self._frames       = []
        self._stop_event   = threading.Event()
        self._rate         = 16000
        self._chunk        = 1024

    def start(self):
        import pyaudio
        self._frames = []
        self._stop_event.clear()
        self.is_recording = True
        def _record():
            pa     = pyaudio.PyAudio()
            stream = pa.open(format=pyaudio.paInt16, channels=1,
                             rate=self._rate, input=True,
                             frames_per_buffer=self._chunk)
            print("  [MIC] Recording...")
            while not self._stop_event.is_set():
                self._frames.append(stream.read(self._chunk, exception_on_overflow=False))
            stream.stop_stream(); stream.close(); pa.terminate()
            print("  [MIC] Stopped")
        threading.Thread(target=_record, daemon=True).start()

    def stop_and_transcribe(self):
        self._stop_event.set()
        self.is_recording = False
        time.sleep(0.3)
        if not self._frames:
            return ""
        import io, wave
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(self._rate)
            wf.writeframes(b''.join(self._frames))
        buf.seek(0)
        recognizer = sr.Recognizer()
        with sr.AudioFile(buf) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"  [STT] Recognized: '{text}'")
            return text
        except:
            return ""

def display_thread(recorder, peer_sock_holder):
    blank_left  = np.zeros((DISPLAY_H, DISPLAY_W // 2, 3), dtype=np.uint8)
    blank_right = np.zeros((DISPLAY_H, DISPLAY_W // 2, 3), dtype=np.uint8)
    blank_left[:] = (30,30,30)
    blank_right[:] = (30,30,30)

    cv2.namedWindow("Live ASL Translator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live ASL Translator", DISPLAY_W, DISPLAY_H + 50)

    while True:
        with state.lock:
            left  = state.my_cam_frame if state.my_cam_frame is not None else blank_left
            right = state.peer_gif_frame if state.peer_gif_frame is not None else blank_right
            connected = state.connected

        if left.shape[0]  != DISPLAY_H: left  = cv2.resize(left,  (DISPLAY_W//2, DISPLAY_H))
        if right.shape[0] != DISPLAY_H: right = cv2.resize(right, (DISPLAY_W//2, DISPLAY_H))

        divider = np.zeros((DISPLAY_H, 4, 3), dtype=np.uint8)
        divider[:] = (0, 220, 255)

        combined = np.hstack([left, divider, right])
        status_bar = np.zeros((50, DISPLAY_W + 4, 3), dtype=np.uint8)
        status_bar[:] = (15,15,15)
        conn_color = (0,200,100) if connected else (0,0,200)
        conn_text  = f"Connected to {state.peer_ip}" if connected else "NOT CONNECTED"
        cv2.putText(status_bar, conn_text, (10, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, conn_color, 2)
        mic_text = "SPACE: start/stop mic  |  C: clear signs  |  Q: quit"
        if recorder.is_recording:
            mic_text = "RECORDING... press SPACE to stop"
        cv2.putText(status_bar, mic_text, (DISPLAY_W//2 - 200, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

        final = np.vstack([combined, status_bar])
        cv2.imshow("Live ASL Translator", final)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'):
            with state.lock:
                state.my_sentence    = []
                state.peer_stop_gif  = True
            print("  Cleared")

        elif key == ord(' '):
            if recorder.is_recording:
                # Stop → transcribe → send to peer
                def process():
                    text = recorder.stop_and_transcribe()
                    if text and peer_sock_holder and state.connected:
                        send_message(peer_sock_holder[0], "speech_text", text)
                        print(f"  [STT] Sent to peer: '{text}'")
                    elif not text:
                        print("  [STT] Nothing recognized")
                threading.Thread(target=process, daemon=True).start()
            else:
                recorder.start()

    cv2.destroyAllWindows()
    os._exit(0)

def main():
    print("=" * 60)
    print("  LIVE BIDIRECTIONAL NETWORKED ASL TRANSLATOR")
    print("  Both machines run this same script")
    print("=" * 60)

    # Network setup
    print("\n  Are you the HOST (waiting) or CLIENT (connecting)?")
    print("  H = Host (run first, share your IP)")
    print("  C = Client (enter host's IP)")

    peer_sock_holder = []

    while True:
        choice = input("  Enter H or C: ").strip().upper()
        if choice == 'H':
            t = threading.Thread(target=start_server, args=(peer_sock_holder,), daemon=True)
            t.start()
            t.join()
            break
        elif choice == 'C':
            peer_ip = input("  Enter host IP address: ").strip()
            start_client(peer_ip, peer_sock_holder)
            break
        else:
            print("  Please enter H or C")

    recv_thread = threading.Thread(
        target=receive_messages,
        args=(peer_sock_holder[0],),
        daemon=True
    )
    recv_thread.start()


    print("\n  Loading LSTM model...")
    model = load_model(MODEL_PATH)
    print("  [OK] Model loaded")

    os.makedirs(os.path.join(GIF_DIR, 'letters'), exist_ok=True)
    os.makedirs(os.path.join(GIF_DIR, 'words'),   exist_ok=True)


    recorder = PushToTalkRecorder()

    threads = [
        threading.Thread(target=sign_detection_thread,
                         args=(model, peer_sock_holder), daemon=True),
        threading.Thread(target=gif_display_thread, daemon=True),
        threading.Thread(target=tts_receiver_thread, daemon=True),
    ]
    for t in threads:
        t.start()

    print("\n  [READY] Live translator running!")
    print("  SPACE = start/stop mic recording")
    print("  C     = clear sign buffer")
    print("  Q     = quit")
    print()
    display_thread(recorder, peer_sock_holder)


if __name__ == '__main__':
    main()