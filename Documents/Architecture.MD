# Architecture — SignBridge

A detailed breakdown of every module, every design decision, and how they connect.

---

## Overview

SignBridge has three layers:

1. **ML Pipeline** — sign detection, normalization, inference, NLP
2. **Web Layer** — Flask-SocketIO backend, WebSocket streaming
3. **Verification Layer** — Llama3 RAG, async sign authentication

All three run simultaneously on a single machine.

---

## ML Pipeline

### Feature Engineering

MediaPipe Holistic extracts 1,662 features per frame:
- 468 face landmarks × 3 = 1,404 dims
- 33 pose landmarks × 4 = 132 dims  
- 21 hand landmarks × 3 × 2 hands = 126 dims

**We keep only the 126 hand dimensions.** Face and pose landmarks introduce noise — signs performed near the face create false correlations with face landmarks. Removing them improved accuracy and eliminated an entire class of false positives.

### Wrist-Origin Normalization

```python
def normalize_hand(hand_63):
    pts   = hand_63.reshape(21, 3)
    wrist = pts[0].copy()
    pts   = pts - wrist                              # Step 1: translate to origin
    max_d = np.linalg.norm(pts, axis=1).max()
    if max_d > 1e-6:
        pts = pts / max_d                            # Step 2: scale invariance
    return pts.flatten()
```

**Step 1 (Translation):** Subtract wrist position from all 21 landmarks. Every coordinate becomes relative to the wrist, not the camera frame.

**Step 2 (Scale):** Divide by the maximum Euclidean distance from wrist to any landmark. The entire hand fits inside a unit sphere.

**Result:** The same hand configuration at 30cm and 100cm from the camera produces vectors with cosine similarity >0.99.

### LSTM Model

```
Input:      (batch, 30, 126)
LSTM:       128 units, return_sequences=True
LSTM:       64 units
Dense:      64 units, ReLU
Dense:      32 units, ReLU
Output:     32 units, Softmax
Parameters: 188,288
```

**Why LSTM over alternatives:**
- **vs RNN:** Vanishing gradients make RNNs unreliable across 30 timesteps
- **vs Transformer:** 960 training sequences is far too small; transformers overfit immediately; O(n²) attention adds latency on CPU
- **vs CNN:** CNNs are spatial, not temporal — they don't capture sign motion across frames

### Inference Pipeline (Translator.py)

Four gates run before any prediction is accepted:

```
Frame in
  │
  ├─ Gate 1: Hand Presence
  │   MediaPipe must detect ≥1 hand
  │
  ├─ Gate 2: Motion Gate
  │   std(keypoints[-6:], axis=0).mean() < 0.012
  │   Hand must be still — no mid-sign predictions
  │
  ├─ Gate 3: Confidence Threshold
  │   Gestures: conf ≥ 0.75
  │   Letters:  conf ≥ 0.92
  │
  ├─ Gate 4: Confidence Gap
  │   top_conf - second_conf ≥ 0.25
  │
  └─ Stability Buffer
      10 consecutive frames unanimous → accept
      3-second cooldown → prevents duplicates
```

### T5 Gloss Module (gloss_t5.py)

Converts raw sign buffer to grammatical English:

```
Buffer: ['H', 'E', 'L', 'L', 'O', 'I LOVE YOU', 'Y', 'O', 'U']
         │
Stage 1: Letter grouping
         ['HELLO', 'I LOVE YOU', 'YOU']
         │
Stage 2: Rule fast path (known combinations)
         Matched → instant output
         │
Stage 3: T5-small generation (unknown combinations)
         Prompt: "convert ASL gloss to English sentence: HELLO I LOVE YOU YOU"
         Output: "Hello, I love you."
```

### Cosine Angle Discriminator (angle_discriminator.py)

For visually similar sign pairs, invoked when:
- Confidence < 0.99, OR
- Confidence gap < 0.50

Computes cosine angle between detected hand vector and precomputed class mean vectors.

| Pair | Angular Gap | Separable |
|------|------------|-----------|
| V vs Z | 22.60° | ✓ |
| N vs J | 12.43° | ✓ |
| O vs E | 6.93°  | ✓ |
| B vs C | 5.22°  | ✓ |
| Yes vs I Love You | 5.58° | ✓ |

---

## Web Layer

### Why WebSocket

HTTP: request → response → close. 30fps video = 30 new TCP connections per second. Impractical.

WebSocket: one persistent bidirectional connection after a single HTTP upgrade handshake. The server pushes frames, detections, verifications — all through the same channel.

### Flask-SocketIO Events

| Event (Client → Server) | Action |
|--------------------------|--------|
| `start_sign_detection` | Spawns sign detection green thread for this client |
| `stop_sign_detection` | Stops detection loop |
| `clear_buffer` | Resets sign buffer |
| `convert_gloss` | Runs T5 gloss on current buffer |
| `speech_to_sign` | Looks up GIFs for text |
| `create_room` | Creates live translation room |
| `join_room_req` | Joins existing room |
| `leave_room_req` | Leaves room, notifies peer |

| Event (Server → Client) | Payload |
|--------------------------|---------|
| `video_frame` | base64 JPEG + word + conf + still + top3 |
| `sign_detected` | word + buffer + conf |
| `gloss_result` | sentence + method |
| `sign_sequence` | GIF sequence array |
| `sign_verified` | RAG verification result |
| `peer_sign` | sentence + from (live rooms) |
| `peer_speech` | text + sequence + from (live rooms) |
| `peer_joined` | name |
| `peer_disconnected` | msg |

### Per-Client Architecture

Each connected browser gets a unique `request.sid`. Sign detection runs as an independent `eventlet.spawn` green thread per client. State is stored in `client_states[sid]`. No two clients interfere.

### Eventlet

Eventlet replaces Python's threading with cooperative green threads:

```python
# Must be first two lines — before all other imports
import eventlet
eventlet.monkey_patch()
```

`eventlet.sleep(0.033)` yields the event loop between frames. `time.sleep()` would block the entire server.

### Live Rooms

```python
live_rooms = {}  # room_code → [sid1, sid2]
```

When a sign is detected, the server iterates the room, skips the sender's SID, and emits `peer_sign` to the peer. The peer's browser receives the sentence, plays GIFs, and speaks it via TTS. Speech-to-sign works identically via `peer_speech`.

---

## Verification Layer

### RAG Pipeline (asl_rag.py)

```
Sign detected
     │
Stage 1 — RETRIEVAL
     Exact match in asl_knowledge_base.json
     Fuzzy fallback (SequenceMatcher ratio > 0.6)
     │
Stage 2 — AUGMENTATION
     Retrieved entry injected into prompt:
     "Sign X detected with Y% confidence.
      Authentic ASL reference: [handshape, movement, location, source]
      Provide a one-sentence verdict: VERIFIED or UNCERTAIN."
     │
Stage 3 — GENERATION
     Llama3 (local via Ollama) returns verdict
     Parsed, emitted via sign_verified WebSocket event
     │
UI panel updates:
     Badge (ASL VERIFIED ✓ / LOW CONFIDENCE / NOT IN KB)
     Handshape, movement, location
     Source citation
     LLM verdict sentence
     Latency
```

**Async execution:** `eventlet.spawn(run_rag, sign, conf, sid)` — never touches the camera loop.

**Fallback:** If Ollama is offline, confidence-threshold-only verification activates. The camera loop is unaffected.

### Knowledge Base Sources

- **ASL University (Lifeprint.com)** — Dr. Bill Vicars — all 32 signs
- **Gallaudet University ASL Connect** — Standard Fingerspelling Chart — A–Z

---

## Data Flow Summary

```
Browser A                    app.py                    Browser B
    │                           │                           │
    │── start_sign_detection ──▶│                           │
    │                           │ eventlet.spawn(loop, sid) │
    │◀── video_frame (30fps) ───│                           │
    │◀── sign_detected ─────────│                           │
    │                           │──── sign_verified ───────▶│  (RAG async)
    │── join_room_req ──────────▶│◀── join_room_req ─────────│
    │                           │ live_rooms[code] = [A, B] │
    │◀── peer_sign ─────────────│◀────── sign detected ──────│
    │    (TTS speaks it)        │                           │
    │── speech_to_sign ─────────▶│────── peer_speech ───────▶│
    │                           │                    (GIFs play)
```
