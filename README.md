# SignBridge

A real-time, two-way American Sign Language translator that runs entirely on a
laptop CPU — no GPU, no cloud services, no special hardware. Just a webcam, a
microphone and a browser.

Signs are recognised from the webcam and spoken aloud; speech is transcribed
locally and played back as sign animations. Both directions work offline.

---

## What it does

**Sign → Speech.** MediaPipe tracks the hands, an LSTM classifies 30-frame
windows into one of 32 signs, a T5 model assembles the recognised signs into a
sentence, and the sentence is spoken.

**Speech → Sign.** Whisper transcribes microphone audio on-device, and the
words are rendered as sign animations.

**Live rooms.** Two people on the same network can join a room code and
translate to each other in real time — one signing, one speaking.

**Sign verification.** Every detection is checked against a knowledge base of
ASL descriptions, with a local LLM confirming the handshape, movement and
location match what the sign should look like. This exists so a detection can
be shown to be a real ASL sign rather than just a confident model output.

### Vocabulary

32 classes: `Hello`, `Thanks`, `Yes`, `I LOVE YOU`, `No`, `Sorry`, and the
letters `A`–`Z`.

---

## Why it works at a distance

The core problem with landmark-based sign recognition is that the same sign
produces completely different coordinates depending on how far the signer
stands from the camera. A model trained at one distance degrades badly at
another.

SignBridge normalizes every hand before it reaches the model:

1. Translate so the wrist (landmark 0) sits at the origin
2. Divide by the largest distance from the wrist to any landmark

What survives is hand *shape*, independent of position and scale. The same
sign at 30 cm, 60 cm and 100 cm yields vectors with cosine similarity above
0.99.

This also cuts the input from MediaPipe Holistic's 1,662 features to 126 —
two hands, 21 landmarks each, three coordinates — by dropping the pose and
face blocks, which carried no signal the hands did not already provide.

The transform lives in one place (`signbridge/`) so the training pipeline, the
web backend and the Android port cannot drift apart.

---

## How a prediction is made

A detection has to clear four gates before it is accepted:

| Gate | Condition |
|---|---|
| Hand presence | At least one hand detected |
| Motion | Landmark standard deviation over 6 frames below 0.012 — the hand has settled |
| Confidence | 75% for gestures, 92% for letters |
| Confidence gap | Top prediction at least 25 points clear of the runner-up |

Past those, a 10-frame stability buffer requires the same sign repeatedly, and
a 3-second cooldown prevents one held sign from firing twice.

Visually similar pairs get an additional check. Signs like V/Z, O/E and V/W
differ by only a few degrees of finger angle, so a cosine-angle discriminator
separates them where softmax confidence alone would not.

### Model

A 2-layer LSTM (128 → 64) with two dense layers, 187,264 parameters, trained
on 960 sequences of 30 frames. Test accuracy is 98.96%.

LSTM rather than a Transformer because the sequences are short and fixed at 30
frames, the dataset is small enough that attention would overfit, and
inference has to stay inside a video frame budget on a CPU.

---

## Running it

### Requirements

- Python 3.11 or 3.12
- A webcam and a microphone
- [Ollama](https://ollama.com) with `llama3` pulled, for sign verification
  (optional — the system falls back to knowledge-base-only verification)

### Setup

```bash
git clone https://github.com/Shashank-Singh-1110/Bi-Directional-Sign-Language-Translator-.git
cd Bi-Directional-Sign-Language-Translator-

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Model weights

The TFLite model (`models/tflite/action_native_fp16.tflite`) is included and
is all the web app needs.

The Keras model (`models/action_norm.h5`) is not in the repository — it is
attached to the latest [Release](../../releases). Download it into `models/`
if you want to retrain, convert, or run the training scripts.

### Start

```bash
python app.py
```

Open <http://localhost:5001>.

On first run Whisper downloads its `small.en` weights (~500 MB), which takes
a couple of minutes. Subsequent starts load in about 4 seconds.

### Live rooms

Both devices must be on the same network. Find your IP:

```bash
ipconfig getifaddr en0        # macOS
hostname -I                   # Linux
```

One person creates a room and shares the code; the other opens
`http://<your-ip>:5001` and joins with it.

---

## Repository layout

```
app.py                   Flask + Socket.IO server; the entry point
index.html               Single-page frontend

signbridge/              Library — importable without a camera or the dataset
  tflite_inference.py      TFLite wrapper, drop-in for the Keras model
  whisper_stt.py           Offline speech-to-text
  ASL_RAG.py               Sign verification against the knowledge base
  Gloss.py                 Signs to sentences
  asl_knowledge_base.json  32 signs: handshape, movement, location, source

scripts/
  data/                  Data collection and normalization
  convert/               TFLite conversion and test fixture export
  demo/                  Standalone desktop demos

models/tflite/           Deployed model and benchmark report
assets/signs_gifs/       Sign animations for speech-to-sign
documents/               Papers, presentations and design notes
```

Not in the repository: `data/` (the recorded dataset) and `.h5` weights. Both
are regenerable — see below.

---

## Rebuilding from scratch

```bash
# 1. Record sequences (webcam, interactive)
python "scripts/data/Data collection.py"

# 2. Normalize and train
python scripts/data/normalize_and_retrain.py

# 3. Convert for mobile
python scripts/convert/convert_to_native_tflite.py
```

---

## Mobile

The model is converted to TensorFlow Lite using builtin operations only, so an
Android build needs just the base runtime rather than the much larger Select
Ops library.

| | Size | Latency (p50) | Agreement with Keras |
|---|---|---|---|
| Keras `.h5` | 2241 KB | 13.3 ms | — |
| TFLite float16 | 386 KB | 0.6 ms | 100.0% |
| TFLite int8 | 209 KB | 0.7 ms | 97.9% |
| **TFLite native fp16** | **392 KB** | **0.38 ms** | **100.0%** |

int8 is smaller but flips predictions on the visually similar pairs, where the
angular margins are only a few degrees — so float16 is what ships.

On-device footprint is 392 KB of model plus roughly 1 MB of runtime.

---

## Authors

**Shashank Singh** and **Bhavya Sharma**
Under the guidance of **Mrs. Archna Lakhe**
Mukesh Patel School of Technology Management & Engineering,
SVKM's NMIMS, Mumbai

---

## Accessibility context

Around 63 million people in India are deaf or hard of hearing, against roughly
300 certified sign language interpreters. Most translation tools require a
network connection, a subscription, or hardware that is not realistic in a
classroom or a clinic.

SignBridge runs offline on an ordinary laptop. That constraint shaped every
technical decision here — the 126-dimension input, the small LSTM, the
on-device speech recognition, the local LLM.

---

## License

MIT — see [LICENSE](LICENSE).
