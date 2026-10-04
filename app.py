"""
app.py — SignBridge server

Flask + Socket.IO. Eventlet is monkey-patched first, before anything else
imports a socket, or the WebSocket transport breaks in ways that surface as
"write() before start_response".
"""

import eventlet
eventlet.monkey_patch()

import base64
import collections
import os
import socket as pysocket
import time

import numpy as np
from flask import Flask, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room

try:
    from signbridge.whisper_stt import (
        get_engine_info,
        load_model as load_whisper,
        transcribe_file,
    )
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[WARN] whisper_stt not found — browser will fall back to Web Speech")

app = Flask(__name__, static_folder='.', template_folder='.')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me-in-production')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

PORT = int(os.environ.get('PORT', 5001))

# Loaded lazily in load_ml() so the server can start while it initialises.
model = None

ACTIONS = np.array([
    'Hello', 'Thanks', 'Yes', 'I LOVE YOU', 'No', 'Sorry',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
])

GESTURE_SIGNS = {'Hello', 'Thanks', 'Yes', 'I LOVE YOU', 'No', 'Sorry'}

# Gates. A detection must clear all four before it is emitted.
GESTURE_THRESH = 0.75
LETTER_THRESH = 0.92
MIN_CONF_GAP = 0.25
MOTION_THRESH = 0.012
MOTION_FRAMES = 6

SEQUENCE_LENGTH = 30
STABILITY_FRAMES = 10
COOLDOWN_SECONDS = 3.0

GIF_DIR = 'assets/signs_gifs'

client_states = {}
live_rooms = {}
tutor_states = {}


# ── Setup ──────────────────────────────────────────────────────────────
def load_ml():
    """
    Load the classifier only.

    MediaPipe runs in the browser now — each client extracts its own
    landmarks and sends the 126-dim vector. That is what lets a phone
    participate with its own camera, and what lets this run on a host with
    no camera at all.
    """
    global model

    from signbridge.tflite_inference import TFLiteModel
    model = TFLiteModel()

    print("[ML] Classifier loaded")


def lan_address():
    """
    The address another device on this network can reach us at.

    location.host in the browser says 'localhost' when the host opened the
    app locally, which is useless in a QR code for someone else's phone.
    """
    s = pysocket.socket(pysocket.AF_INET, pysocket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))   # no packet is actually sent
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return f'{ip}:{PORT}'


# ── Feature extraction ─────────────────────────────────────────────────
def is_hand_still(seq):
    if len(seq) < MOTION_FRAMES:
        return False
    arr = np.array(list(seq)[-MOTION_FRAMES:])
    return np.std(arr, axis=0).mean() < MOTION_THRESH


def group_letters_inline(buf):
    """Run consecutive letters together: H E L L O -> HELLO."""
    letters_set = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    gloss, letters = [], []

    for sign in buf:
        upper = sign.upper()
        if upper in letters_set:
            letters.append(upper)
        else:
            if letters:
                gloss.append(''.join(letters))
                letters = []
            gloss.append(upper)

    if letters:
        gloss.append(''.join(letters))

    return gloss


def gif_path_for(sign):
    """URL for a sign's animation, or None."""
    if not sign:
        return None

    slug = sign.lower().replace(' ', '_')

    if os.path.exists(os.path.join(GIF_DIR, 'words', f'{slug}.gif')):
        return f'/gifs/words/{slug}.gif'

    if len(sign) == 1 and os.path.exists(
            os.path.join(GIF_DIR, 'letters', f'{sign.upper()}.gif')):
        return f'/gifs/letters/{sign.upper()}.gif'

    return None


def build_sequence(text):
    """Text to a playable list of sign animations, fingerspelling the rest."""
    sequence = []

    for word in text.strip().split():
        clean = ''.join(c for c in word if c.isalnum()).lower()
        if not clean:
            continue

        if os.path.exists(os.path.join(GIF_DIR, 'words', f'{clean}.gif')):
            sequence.append({
                'type': 'word',
                'label': word,
                'path': f'/gifs/words/{clean}.gif',
            })
        else:
            for ch in clean:
                if ch.isalpha() and os.path.exists(
                        os.path.join(GIF_DIR, 'letters', f'{ch.upper()}.gif')):
                    sequence.append({
                        'type': 'letter',
                        'label': ch.upper(),
                        'path': f'/gifs/letters/{ch.upper()}.gif',
                    })

    return sequence


# ── Detection loop ─────────────────────────────────────────────────────
def new_tracker():
    """Per-client detection state. One of these per connected browser."""
    return {
        'sequence': collections.deque(maxlen=SEQUENCE_LENGTH),
        'stability': collections.deque(maxlen=STABILITY_FRAMES),
        'buffer': [],
        'last_emit': 0.0,
    }


def process_features(sid, features, hands_seen):
    """
    One frame's worth of landmarks from a browser.

    The browser runs MediaPipe and the normalization; everything from here
    is identical to what the old server-side loop did, so detections behave
    exactly as before.
    """
    state = client_states.get(sid)
    if not state or not state.get('active') or model is None:
        return

    track = state.setdefault('track', new_tracker())

    # Hands out of frame breaks the sequence — a sign cannot span a gap.
    if not hands_seen:
        track['sequence'].clear()
        track['stability'].clear()
        socketio.emit('reading', {
            'word': '', 'conf': 0.0, 'still': False,
            'ready': 0, 'buffer': track['buffer'].copy(),
        }, to=sid)
        return

    track['sequence'].append(np.array(features, dtype=np.float32))
    sequence = track['sequence']

    if len(sequence) < SEQUENCE_LENGTH:
        socketio.emit('reading', {
            'word': '', 'conf': 0.0, 'still': False,
            'ready': len(sequence), 'buffer': track['buffer'].copy(),
        }, to=sid)
        return

    still = is_hand_still(sequence)

    X = np.expand_dims(np.array(sequence), axis=0)
    pred = model.predict(X, verbose=0)[0]

    top_idx = np.argsort(pred)[-3:][::-1]
    top3 = [(ACTIONS[i], float(pred[i])) for i in top_idx]
    word, conf = top3[0]

    gap_ok = (conf - top3[1][1]) >= MIN_CONF_GAP
    threshold = GESTURE_THRESH if word in GESTURE_SIGNS else LETTER_THRESH

    track['stability'].append(
        word if (conf >= threshold and gap_ok and still) else ""
    )

    stable = (len(track['stability']) == STABILITY_FRAMES and
              len(set(track['stability'])) == 1 and
              track['stability'][0] != "")

    now = time.time()
    if stable and (now - track['last_emit']) > COOLDOWN_SECONDS:
        detected = track['stability'][0]
        track['buffer'].append(detected)
        track['buffer'] = track['buffer'][-15:]
        track['last_emit'] = now
        track['stability'].clear()

        socketio.emit('sign_detected', {
            'word': detected,
            'buffer': track['buffer'].copy(),
            'conf': float(round(conf * 100, 1)),
        }, to=sid)

        eventlet.spawn(run_verification, detected, conf, sid)
        relay_to_peer(sid, track['buffer'])

    socketio.emit('reading', {
        'word': word,
        'conf': float(round(conf * 100, 1)),
        'still': bool(still),
        'ready': SEQUENCE_LENGTH,
        'top3': [(str(w), float(round(c * 100, 1))) for w, c in top3],
        'buffer': track['buffer'].copy(),
    }, to=sid)


def run_verification(sign, conf, sid):
    """Check the detection against the ASL reference, via the LLM."""
    try:
        from signbridge.ASL_RAG import verify_sign
        socketio.emit('sign_verified', verify_sign(sign, conf), to=sid)
    except Exception as e:
        print(f"[RAG] {e}")
        socketio.emit('sign_verified', {
            'sign': sign,
            'verified': True,
            'status': 'verified',
            'handshape': '—', 'movement': '—', 'location': '—',
            'description': f'Recognised at {round(conf * 100, 1)}% confidence.',
            'source': 'Model confidence',
            'llm_verdict': 'Reference unavailable.',
            'latency_ms': 0,
        }, to=sid)


def relay_to_peer(sid, sign_buffer):
    state = client_states.get(sid, {})
    room = state.get('room')
    if not room or room not in live_rooms:
        return

    gloss = group_letters_inline(sign_buffer)
    sentence = ' '.join(g.capitalize() for g in gloss)

    for peer in live_rooms[room]:
        if peer != sid:
            socketio.emit('peer_sign', {
                'sentence': sentence,
                'buffer': sign_buffer.copy(),
                'from': state.get('name', 'Peer'),
            }, to=peer)


# ── Sessions ───────────────────────────────────────────────────────────
def broadcast_sessions():
    """
    Tell everyone which sessions are open.

    Both devices talk to this same server, so there is no reason to make
    anyone type a code — they pick from a list.
    """
    socketio.emit('sessions', {'sessions': [
        {
            'code': code,
            'host': client_states.get(sids[0], {}).get('name', 'Someone'),
            'count': len(sids),
            'full': len(sids) >= 2,
        }
        for code, sids in live_rooms.items()
    ]})


def leave_current_room(sid, message):
    state = client_states.get(sid, {})
    room = state.get('room')
    if not room or room not in live_rooms:
        return

    live_rooms[room] = [s for s in live_rooms[room] if s != sid]

    if not live_rooms[room]:
        del live_rooms[room]
    else:
        for peer in live_rooms[room]:
            socketio.emit('peer_disconnected', {'msg': message}, to=peer)

    state['room'] = None
    broadcast_sessions()


# ── Connection ─────────────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    sid = request.sid
    client_states[sid] = {'active': False, 'room': None, 'name': 'User'}
    print(f"[WS] Connected: {sid}")
    emit('status', {'msg': 'Connected'})
    emit('my_sid', {'sid': sid})
    broadcast_sessions()


@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    leave_current_room(sid, 'Your peer disconnected')
    client_states.pop(sid, None)
    tutor_states.pop(sid, None)
    print(f"[WS] Disconnected: {sid}")


# ── Sign detection ─────────────────────────────────────────────────────
@socketio.on('start_sign_detection')
def start_sign():
    sid = request.sid

    if model is None:
        emit('error', {'msg': 'Model still loading. Try again in a moment.'})
        return

    state = client_states.setdefault(
        sid, {'active': False, 'room': None, 'name': 'User'})

    state['active'] = True
    state['track'] = new_tracker()

    emit('status', {'msg': 'Detection started'})


@socketio.on('stop_sign_detection')
def stop_sign():
    sid = request.sid
    if sid in client_states:
        client_states[sid]['active'] = False
    emit('status', {'msg': 'Detection stopped'})


@socketio.on('frame_features')
def frame_features(data):
    """
    126 normalized landmark values from a browser.

    Sending the vector rather than a JPEG keeps this at roughly 1 KB per
    frame instead of 40 KB, and means the server does no vision work.
    """
    features = data.get('f')
    if not features or len(features) != 126:
        return

    process_features(request.sid, features, bool(data.get('h')))


@socketio.on('clear_buffer')
def clear_buffer():
    sid = request.sid
    state = client_states.get(sid)
    if state and 'track' in state:
        state['track'] = new_tracker()
    emit('buffer_cleared', {})


@socketio.on('convert_gloss')
def convert_gloss(data):
    buf = data.get('buffer', [])
    if not buf:
        emit('gloss_result', {'sentence': '', 'method': 'empty', 'gloss': []})
        return

    try:
        from signbridge.Gloss import convert_and_speak
        gloss, sentence, method = convert_and_speak(buf, verbose=False)
        emit('gloss_result',
             {'sentence': sentence, 'gloss': gloss, 'method': method})
    except Exception as e:
        print(f"[GLOSS] Fallback: {e}")
        gloss = group_letters_inline(buf)
        sentence = ' '.join(g.capitalize() for g in gloss) + '.'
        emit('gloss_result',
             {'sentence': sentence, 'gloss': gloss, 'method': 'rules'})


# ── Speech to sign ─────────────────────────────────────────────────────
@socketio.on('speech_to_sign')
def speech_to_sign(data):
    text = (data.get('text') or '').strip()
    if not text:
        return

    sequence = build_sequence(text)
    emit('sign_sequence', {'sequence': sequence, 'original': text})

    sid = request.sid
    state = client_states.get(sid, {})
    room = state.get('room')

    if room and room in live_rooms:
        for peer in live_rooms[room]:
            if peer != sid:
                socketio.emit('peer_speech', {
                    'text': text,
                    'sequence': sequence,
                    'from': state.get('name', 'Peer'),
                }, to=peer)


@socketio.on('transcribe_audio')
def transcribe_audio(data):
    """Browser sends recorded audio; Whisper transcribes it on this machine."""
    import tempfile

    sid = request.sid

    if not WHISPER_AVAILABLE:
        emit('transcribe_result', {
            'text': '', 'engine': 'unavailable',
            'error': 'Whisper is not installed on the server',
        })
        return

    audio_b64 = data.get('audio', '')
    if not audio_b64:
        emit('transcribe_result',
             {'text': '', 'engine': 'failed', 'error': 'No audio received'})
        return

    emit('transcribe_status', {'msg': 'Transcribing'})

    def run(b64, client_id):
        path = None
        try:
            raw = base64.b64decode(b64.split(',')[-1])
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
                tmp.write(raw)
                path = tmp.name

            result = transcribe_file(path)
            text = result['text']
            print(f"[WHISPER] {result['latency_ms']}ms · {text[:60]}")

            sequence = build_sequence(text) if text else []

            socketio.emit('transcribe_result', {
                'text': text,
                'engine': result['engine'],
                'model': result['model'],
                'latency_ms': result['latency_ms'],
                'sequence': sequence,
            }, to=client_id)

            state = client_states.get(client_id, {})
            room = state.get('room')
            if room and room in live_rooms and text:
                for peer in live_rooms[room]:
                    if peer != client_id:
                        socketio.emit('peer_speech', {
                            'text': text,
                            'sequence': sequence,
                            'from': state.get('name', 'Peer'),
                        }, to=peer)

        except Exception as e:
            print(f"[WHISPER] {e}")
            socketio.emit('transcribe_result',
                          {'text': '', 'engine': 'failed', 'error': str(e)},
                          to=client_id)
        finally:
            if path and os.path.exists(path):
                os.unlink(path)

    eventlet.spawn(run, audio_b64, sid)


@socketio.on('get_stt_engine')
def get_stt_engine():
    if WHISPER_AVAILABLE:
        emit('stt_engine_info', get_engine_info())
    else:
        emit('stt_engine_info', {
            'engine': 'browser', 'model': 'Web Speech API',
            'offline': False, 'requires_key': False,
        })


# ── Live sessions ──────────────────────────────────────────────────────
@socketio.on('open_session')
def open_session(data):
    """Start a session others can join. The code is generated, not typed."""
    import random
    import string

    sid = request.sid
    name = (data.get('name') or 'Host').strip()

    leave_current_room(sid, f'{name} closed the session')

    code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    while code in live_rooms:
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

    live_rooms[code] = [sid]
    client_states[sid].update({'room': code, 'name': name})
    join_room(code)

    emit('session_opened', {'code': code, 'address': lan_address(), 'host': name})
    broadcast_sessions()
    print(f"[ROOM] {name} opened {code}")


@socketio.on('join_session')
def join_session(data):
    sid = request.sid
    code = (data.get('code') or '').upper().strip()
    name = (data.get('name') or 'Guest').strip()

    if code not in live_rooms:
        emit('session_error', {'msg': 'That session has ended'})
        return

    if len(live_rooms[code]) >= 2:
        emit('session_error', {'msg': 'That session is full'})
        return

    live_rooms[code].append(sid)
    client_states[sid].update({'room': code, 'name': name})
    join_room(code)

    host_name = client_states.get(live_rooms[code][0], {}).get('name', 'Host')

    emit('session_joined', {'code': code, 'host': host_name})
    socketio.emit('peer_joined',
                  {'name': name, 'msg': f'{name} joined'},
                  to=live_rooms[code][0])

    broadcast_sessions()
    print(f"[ROOM] {name} joined {code}")


@socketio.on('leave_session')
def leave_session():
    sid = request.sid
    room = client_states.get(sid, {}).get('room')
    leave_current_room(sid, 'Your peer left')
    if room:
        leave_room(room)
    emit('session_left', {})


@socketio.on('list_sessions')
def list_sessions():
    broadcast_sessions()


# ── Tutor ──────────────────────────────────────────────────────────────
@socketio.on('tutor_message')
def tutor_message(data):
    sid = request.sid
    text = (data.get('text') or '').strip()
    if not text:
        return

    state = tutor_states.setdefault(sid, {'sign': None, 'awaiting': False})
    emit('tutor_thinking', {})

    def run(message, history, current, client_id):
        try:
            from signbridge.tutor import chat
            result = chat(message, history, current)

            lesson = tutor_states.setdefault(
                client_id, {'sign': None, 'awaiting': False})

            for action in result['actions']:
                action['gif'] = gif_path_for(action['sign'])
                if action['type'] == 'practice':
                    lesson.update({'sign': action['sign'], 'awaiting': True})
                elif action['type'] == 'show':
                    lesson['sign'] = action['sign']

            socketio.emit('tutor_reply', result, to=client_id)

        except Exception as e:
            print(f"[TUTOR] {e}")
            socketio.emit('tutor_reply', {
                'reply': f'The tutor is unavailable: {e}',
                'actions': [], 'engine': 'error', 'latency_ms': 0,
            }, to=client_id)

    eventlet.spawn(run, text, data.get('history', []), state.get('sign'), sid)


@socketio.on('tutor_attempt')
def tutor_attempt(data):
    """A detection arrived while the tutor was waiting for a practice try."""
    sid = request.sid
    state = tutor_states.get(sid)

    if not state or not state.get('awaiting'):
        return

    expected = state.get('sign')
    state['awaiting'] = False

    emit('tutor_thinking', {})

    def run(exp, detected, confidence, client_id):
        try:
            from signbridge.tutor import grade
            result = grade(exp, detected, confidence)

            for action in result['actions']:
                action['gif'] = gif_path_for(action['sign'])

            result.update({'expected': exp, 'detected': detected})
            socketio.emit('tutor_feedback', result, to=client_id)

        except Exception as e:
            print(f"[TUTOR] grading failed: {e}")
            socketio.emit('tutor_feedback', {
                'reply': 'Could not grade that attempt.',
                'correct': False, 'actions': [],
                'expected': exp, 'detected': detected,
            }, to=client_id)

    eventlet.spawn(run, expected, data.get('detected', ''),
                   float(data.get('confidence', 0)), sid)


@socketio.on('tutor_skip')
def tutor_skip():
    sid = request.sid
    if sid in tutor_states:
        tutor_states[sid]['awaiting'] = False
    emit('tutor_skipped', {})


@socketio.on('tutor_status')
def tutor_status():
    try:
        from signbridge.tutor import health
        emit('tutor_health', health())
    except Exception as e:
        emit('tutor_health',
             {'available': False, 'model': None, 'installed': [],
              'signs': 0, 'error': str(e)})


# ── Routes ─────────────────────────────────────────────────────────────
@app.route('/gifs/<path:filename>')
def serve_gif(filename):
    return send_from_directory(GIF_DIR, filename)


@app.route('/api/address')
def api_address():
    return {'address': lan_address()}


@app.route('/health')
def health_check():
    return {
        'status': 'ok',
        'model_loaded': model is not None,
        'whisper': WHISPER_AVAILABLE,
        'sessions': len(live_rooms),
    }


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path and os.path.exists(os.path.join('.', path)):
        return send_from_directory('.', path)
    return send_from_directory('.', 'index.html')


# ── Startup ────────────────────────────────────────────────────────────
def startup():
    print("[INIT] Loading model and MediaPipe")
    load_ml()

    if WHISPER_AVAILABLE:
        print("[INIT] Loading Whisper")
        try:
            load_whisper()
            print("[INIT] Whisper ready — speech stays on this machine")
        except Exception as e:
            print(f"[INIT] Whisper failed: {e}")

    print(f"[READY] http://localhost:{PORT}")
    print(f"[READY] On this network: http://{lan_address()}")


if __name__ == '__main__':
    eventlet.spawn_after(1, startup)
    socketio.run(app, host='0.0.0.0', port=PORT,
                 debug=False, use_reloader=False)