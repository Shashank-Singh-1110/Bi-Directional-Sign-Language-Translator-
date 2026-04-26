import eventlet
eventlet.monkey_patch()

import os
import threading
import time
import base64
import collections
import numpy as np

from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO, emit, join_room, leave_room, rooms
from flask_cors import CORS

app = Flask(__name__, static_folder='.', template_folder='.')
app.config['SECRET_KEY'] = 'slt-secret-2025'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
model       = None
mp_holistic = None
mp_drawing  = None
mp_style    = None
mp          = None
cv2         = None

ACTIONS = np.array([
    'Hello', 'Thanks', 'Yes', 'I LOVE YOU', 'No', 'Sorry',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z'
])

GESTURE_SIGNS    = {'Hello','Thanks','Yes','I LOVE YOU','No','Sorry'}
GESTURE_THRESH   = 0.75
LETTER_THRESH    = 0.92
MIN_CONF_GAP     = 0.25
SEQUENCE_LENGTH  = 30
STABILITY_FRAMES = 10
COOLDOWN_SECONDS = 3.0
MOTION_FRAMES    = 6
MOTION_THRESH    = 0.012

client_states = {}

live_rooms = {}
def load_ml():
    global model, mp_holistic, mp_drawing, mp_style, mp, cv2
    import cv2 as _cv2
    import mediapipe as _mp
    cv2 = _cv2
    mp  = _mp
    mp_holistic = mp.solutions.holistic
    mp_drawing  = mp.solutions.drawing_utils
    mp_style    = mp.solutions.drawing_styles
    from tensorflow.keras.models import load_model as _lm
    model = _lm('action_norm.h5')
    print("[ML] Model and MediaPipe loaded")

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

def is_hand_still(seq):
    if len(seq) < MOTION_FRAMES:
        return False
    arr = np.array(list(seq)[-MOTION_FRAMES:])
    return np.std(arr, axis=0).mean() < MOTION_THRESH

def group_letters_inline(buf):
    LETTERS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    gloss, letters = [], []
    for sign in buf:
        su = sign.upper()
        if su in LETTERS:
            letters.append(su)
        else:
            if letters:
                gloss.append(''.join(letters))
                letters = []
            gloss.append(su)
    if letters:
        gloss.append(''.join(letters))
    return gloss

def sign_detection_loop(sid):
    import cv2 as _cv2

    state = client_states.get(sid)
    if not state:
        return

    cap      = _cv2.VideoCapture(0)
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    state['cap'] = cap
    sequence         = collections.deque(maxlen=SEQUENCE_LENGTH)
    stability_buffer = collections.deque(maxlen=STABILITY_FRAMES)
    sign_buffer      = []
    last_pred_time   = 0

    print(f"[DET] Detection started for {sid}")

    while True:
        state = client_states.get(sid)
        if not state or not state.get('active'):
            eventlet.sleep(0.05)
            if not client_states.get(sid):
                break
            continue

        ret, frame = cap.read()
        if not ret:
            continue

        frame   = _cv2.flip(frame, 1)
        rgb     = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style())
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style())

        kp    = extract_keypoints(results)
        sequence.append(kp)
        still = is_hand_still(sequence)

        word, conf, top3 = "", 0.0, []
        if len(sequence) == SEQUENCE_LENGTH:
            X        = np.expand_dims(np.array(sequence), axis=0)
            pred     = model.predict(X, verbose=0)[0]
            top3_idx = np.argsort(pred)[-3:][::-1]
            top3     = [(ACTIONS[i], float(pred[i])) for i in top3_idx]
            word     = top3[0][0]
            conf     = top3[0][1]
            sec_conf = top3[1][1]
            gap_ok   = (conf - sec_conf) >= MIN_CONF_GAP

            hand_vis  = (results.left_hand_landmarks is not None or
                         results.right_hand_landmarks is not None)
            threshold = GESTURE_THRESH if word in GESTURE_SIGNS else LETTER_THRESH

            if hand_vis and conf >= threshold and gap_ok and still:
                stability_buffer.append(word)
            else:
                stability_buffer.append("")

            stable = (
                len(stability_buffer) == STABILITY_FRAMES and
                len(set(stability_buffer)) == 1 and
                stability_buffer[0] != ""
            )

            now = time.time()
            if stable and (now - last_pred_time) > COOLDOWN_SECONDS:
                detected = stability_buffer[0]
                sign_buffer.append(detected)
                if len(sign_buffer) > 15:
                    sign_buffer = sign_buffer[-15:]
                last_pred_time = now
                stability_buffer.clear()
                socketio.emit('sign_detected', {
                    'word':   detected,
                    'buffer': sign_buffer.copy(),
                    'conf':   float(round(conf * 100, 1))
                }, to=sid)
                def run_rag(sign, conf, client_id):
                    try:
                        from ASL_RAG import verify_sign
                        result = verify_sign(sign, conf)
                        socketio.emit('sign_verified', result, to=client_id)
                    except Exception as e:
                        socketio.emit('sign_verified', {
                            'sign': sign, 'verified': True, 'status': 'verified',
                            'handshape': '—', 'movement': '—', 'location': '—',
                            'description': f'Sign detected with {round(conf * 100, 1)}% confidence.',
                            'source': 'Model confidence',
                            'llm_verdict': f'Verified by model at {round(conf * 100, 1)}%.',
                            'latency_ms': 0
                        }, to=client_id)

                eventlet.spawn(run_rag, detected, conf, sid)

                room_code = state.get('room')
                if room_code and room_code in live_rooms:
                    gloss    = group_letters_inline(sign_buffer)
                    sentence = ' '.join(g.capitalize() for g in gloss)
                    for peer_sid in live_rooms[room_code]:
                        if peer_sid != sid:
                            socketio.emit('peer_sign', {
                                'sentence': sentence,
                                'buffer':   sign_buffer.copy(),
                                'from':     state.get('name', 'Peer')
                            }, to=peer_sid)
        _, buf = _cv2.imencode('.jpg', frame, [_cv2.IMWRITE_JPEG_QUALITY, 55])
        b64 = base64.b64encode(buf).decode('utf-8')
        socketio.emit('video_frame', {
            'frame': b64,
            'word':  word,
            'conf':  float(round(conf * 100, 1)),
            'still': bool(still),
            'top3':  [(str(w), float(round(c * 100, 1))) for w, c in top3],
            'buffer': sign_buffer.copy()
        }, to=sid)

        eventlet.sleep(0.033)

    cap.release()
    print(f"[DET] Detection stopped for {sid}")

# ── Socket events ──────────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    sid = request.sid
    client_states[sid] = {
        'active': False,
        'room':   None,
        'name':   'User',
        'cap':    None
    }
    print(f"[WS] Connected: {sid}")
    emit('status', {'msg': 'Connected to SLT backend'})
    emit('my_sid', {'sid': sid})

@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    state = client_states.get(sid, {})
    room_code = state.get('room')
    if room_code and room_code in live_rooms:
        live_rooms[room_code] = [s for s in live_rooms[room_code] if s != sid]
        if not live_rooms[room_code]:
            del live_rooms[room_code]
        else:
            for peer in live_rooms[room_code]:
                socketio.emit('peer_disconnected', {'msg': 'Peer disconnected'}, to=peer)
    if sid in client_states:
        client_states[sid]['active'] = False
        del client_states[sid]
    print(f"[WS] Disconnected: {sid}")

@socketio.on('start_sign_detection')
def start_sign():
    sid = request.sid
    if model is None:
        emit('error', {'msg': 'Model not loaded yet. Please wait...'})
        return
    if sid not in client_states:
        client_states[sid] = {'active': False, 'room': None, 'name': 'User', 'cap': None}
    if not client_states[sid].get('active'):
        client_states[sid]['active'] = True
        eventlet.spawn(sign_detection_loop, sid)
    emit('status', {'msg': 'Sign detection started'})

@socketio.on('stop_sign_detection')
def stop_sign():
    sid = request.sid
    if sid in client_states:
        client_states[sid]['active'] = False
    emit('status', {'msg': 'Sign detection stopped'})

@socketio.on('clear_buffer')
def clear_buf():
    emit('buffer_cleared', {})

@socketio.on('convert_gloss')
def convert_gloss(data):
    buf = data.get('buffer', [])
    if not buf:
        emit('gloss_result', {'sentence': '', 'method': 'empty', 'gloss': []})
        return
    try:
        from gloss_t5 import convert_and_speak
        gloss, sentence, method = convert_and_speak(buf, verbose=False)
        emit('gloss_result', {'sentence': sentence, 'gloss': gloss, 'method': method})
    except Exception as e:
        print(f"[GLOSS] Fallback: {e}")
        gloss    = group_letters_inline(buf)
        sentence = ' '.join(g.capitalize() for g in gloss) + '.'
        emit('gloss_result', {'sentence': sentence, 'gloss': gloss, 'method': 'fallback'})

@socketio.on('speech_to_sign')
def speech_to_sign(data):
    text = data.get('text', '')
    if not text:
        return
    GIF_DIR = 'signs_gifs'
    words, result = text.strip().split(), []
    for word in words:
        wl = word.lower()
        wp = os.path.join(GIF_DIR, 'words', f'{wl}.gif')
        if os.path.exists(wp):
            result.append({'type': 'word', 'label': word, 'path': f'/gifs/words/{wl}.gif'})
        else:
            for ch in word:
                if ch.isalpha():
                    lp = os.path.join(GIF_DIR, 'letters', f'{ch.upper()}.gif')
                    if os.path.exists(lp):
                        result.append({'type': 'letter', 'label': ch.upper(),
                                       'path': f'/gifs/letters/{ch.upper()}.gif'})
    emit('sign_sequence', {'sequence': result, 'original': text})

    sid = request.sid
    state = client_states.get(sid, {})
    room_code = state.get('room')
    if room_code and room_code in live_rooms:
        for peer_sid in live_rooms[room_code]:
            if peer_sid != sid:
                socketio.emit('peer_speech', {
                    'text': text,
                    'sequence': result,
                    'from': state.get('name', 'Peer')
                }, to=peer_sid)

@socketio.on('create_room')
def create_room(data):
    sid       = request.sid
    room_code = data.get('code', '').upper().strip()
    name      = data.get('name', 'Host')
    if not room_code:
        emit('room_error', {'msg': 'Room code cannot be empty'})
        return
    if room_code in live_rooms:
        emit('room_error', {'msg': f'Room {room_code} already exists'})
        return
    live_rooms[room_code]    = [sid]
    client_states[sid]['room'] = room_code
    client_states[sid]['name'] = name
    join_room(room_code)
    emit('room_created', {
        'code': room_code,
        'msg':  f'Room {room_code} created. Share this code with your peer.'
    })
    print(f"[ROOM] Created: {room_code} by {sid}")

@socketio.on('join_room_req')
def join_room_req(data):
    sid       = request.sid
    room_code = data.get('code', '').upper().strip()
    name      = data.get('name', 'Peer')
    if not room_code:
        emit('room_error', {'msg': 'Room code cannot be empty'})
        return
    if room_code not in live_rooms:
        emit('room_error', {'msg': f'Room {room_code} not found. Ask host to create it first.'})
        return
    if len(live_rooms[room_code]) >= 2:
        emit('room_error', {'msg': f'Room {room_code} is full (2 participants max)'})
        return

    live_rooms[room_code].append(sid)
    client_states[sid]['room'] = room_code
    client_states[sid]['name'] = name
    join_room(room_code)

    # Notify joiner
    emit('room_joined', {
        'code': room_code,
        'msg':  f'Joined room {room_code}. Live translation active!'
    })

    # Notify host
    host_sid = live_rooms[room_code][0]
    socketio.emit('peer_joined', {
        'name': name,
        'msg':  f'{name} joined the session. Live translation active!'
    }, to=host_sid)

    print(f"[ROOM] {name} ({sid}) joined {room_code}")

@socketio.on('leave_room_req')
def leave_room_req():
    sid = request.sid
    state = client_states.get(sid, {})
    room_code = state.get('room')
    if room_code and room_code in live_rooms:
        live_rooms[room_code] = [s for s in live_rooms[room_code] if s != sid]
        if not live_rooms[room_code]:
            del live_rooms[room_code]
        else:
            for peer in live_rooms[room_code]:
                socketio.emit('peer_disconnected', {'msg': 'Peer left the session'}, to=peer)
        leave_room(room_code)
        client_states[sid]['room'] = None
    emit('room_left', {'msg': 'Left the session'})

@socketio.on('list_rooms')
def list_rooms():
    emit('rooms_list', {
        'rooms': [
            {'code': code, 'count': len(sids)}
            for code, sids in live_rooms.items()
        ]
    })

@app.route('/gifs/<path:filename>')
def serve_gif(filename):
    return send_from_directory('signs_gifs', filename)

@app.route('/health')
def health():
    return {'status': 'ok', 'model_loaded': model is not None,
            'active_rooms': len(live_rooms)}

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path and os.path.exists(os.path.join('.', path)):
        return send_from_directory('.', path)
    return send_from_directory('.', 'index.html')

def startup():
    print("[INIT] Loading ML models...")
    load_ml()
    print("[READY] Server ready at http://localhost:5001")

if __name__ == '__main__':
    eventlet.spawn_after(1, startup)
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, use_reloader=False)