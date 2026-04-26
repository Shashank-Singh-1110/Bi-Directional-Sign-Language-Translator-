# 🤟 SignBridge — Real-Time Bidirectional ASL Translator

> A real-time, two-way American Sign Language translator that runs entirely on a laptop CPU — no GPU, no special hardware, just a webcam and a browser.

---

## What it does

| Direction | How |
|-----------|-----|
| ✋ Sign → Speech | Webcam detects ASL signs → LSTM model → T5 NLP → spoken English |
| 🎤 Speech → Sign | Speak or type → Google STT → ASL GIF animations |
| 🌐 Live Network | Two browsers, one WiFi → real-time peer translation via Socket.IO rooms |
| 🔍 Verification | Every detected sign verified against ASL University sources via Llama3 RAG |

---

## Demo

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/signbridge-asl-translator.git
cd signbridge-asl-translator

# 2. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate       # Mac/Linux
.venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python app.py

# 5. Open browser
open http://localhost:5001
```

For the live network demo — get your IP and share it with a peer:
```bash
ipconfig getifaddr en0          # Mac
hostname -I                     # Linux
```
Peer opens `http://YOUR_IP:5001` — no setup needed on their machine.

---

## Project Structure

```
signbridge-asl-translator/
├── app.py                    # Flask-SocketIO backend — main server
├── index.html                # SignBridge web UI — 4 tabs
├── Translator.py             # Standalone sign detection (OpenCV window)
├── gloss_t5.py               # T5-small gloss → English conversion
├── Speech_TO_Sign.py         # Push-to-talk speech → ASL GIFs
├── live_translator.py        # TCP peer-to-peer live translation
├── angle_discriminator.py    # Cosine angle profiles for similar signs
├── asl_rag.py                # Llama3 RAG sign verification
├── asl_knowledge_base.json   # 32-sign ASL reference (ASL Univ + Gallaudet)
├── data_collection.py        # Training data recording script
├── normalize_and_retrain.py  # Normalization + multi-run LSTM training
├── signs_gifs/               # ASL GIF library
│   ├── letters/              # A.gif — Z.gif
│   └── words/                # hello.gif, thanks.gif, etc.
└── docs/
    ├── ARCHITECTURE.md       # Full system design
    ├── NORMALIZATION.md      # Domain gap solution explained
    └── DEMO_GUIDE.md         # Step-by-step demo instructions
```

---

## System Architecture

```
SIGN TO SPEECH
Webcam → MediaPipe Holistic → Wrist-Origin Normalization (126-dim)
       → LSTM (32 classes) → 4 Gates → Stability Buffer
       → T5 Gloss → TTS

SPEECH TO SIGN
Mic (push-to-talk) → PyAudio → Google STT
                   → GIF Lookup → ASL GIF Display / Fingerspelling

WEB LAYER
Flask-SocketIO → SignBridge UI (index.html)
              → Live Network Rooms (Socket.IO)
              → Llama3 RAG Verification (async)
```

---

## Key Technical Contributions

**1. Wrist-Origin Normalization**  
Reduces MediaPipe's 1,662-dim output to 126-dim hand-only features, then normalizes to wrist-origin with scale invariance. Solves the domain gap — same hand shape produces identical features at any camera distance.

**2. Four-Gate Inference Pipeline**  
Hand presence → Motion gate → Confidence threshold → Confidence gap → 10-frame stability buffer → 3-second cooldown. Near-zero false positives in real-world conditions.

**3. T5 Gloss Module**  
Converts fingerspelled letter sequences into grammatical English via letter grouping → rule fast path → T5-small beam search.

**4. Networked Live Translation**  
Two browsers on the same network join a Socket.IO room. Signs detected on one machine are forwarded as sentences to the peer's screen and spoken via TTS. First such networked ASL system in published literature.

**5. LLM Sign Verification (RAG)**  
Llama3 running locally via Ollama verifies every detected sign against ASL University and Gallaudet reference descriptions. Runs asynchronously — never blocks the camera loop.

---

## Hardware & Dependencies

- **Tested on:** MacBook Air M1/M5, Python 3.12
- **No GPU required** — all inference runs on CPU

### Core dependencies
```
tensorflow==2.16.2
mediapipe==0.10.14
protobuf==4.25.9
flask
flask-socketio
flask-cors
eventlet
opencv-python
pyaudio
SpeechRecognition
transformers
torch
```

### For LLM verification (optional)
```bash
# Install Ollama
brew install ollama          # Mac
# Pull Llama3
ollama pull llama3
# Run before starting app.py
ollama run llama3
```

---

## Apple Silicon (M1/M2/M3/M4/M5) Setup Notes

```bash
# Install native FLAC (required for SpeechRecognition)
brew install flac

# Install PortAudio (required for PyAudio)
brew install portaudio

# Pin setuptools to avoid distutils conflict
pip install "setuptools==67.8.0"
```

---

## ASL Classes (32 total)

**Gestures (6):** Hello, Thanks, Yes, No, Sorry, I Love You  
**Alphabet (26):** A B C D E F G H I J K L M N O P Q R S T U V W X Y Z

---

## Model

The trained model (`action_norm.h5`) is available in [Releases](../../releases).

- Architecture: 2-layer LSTM (128 → 64 units) + 2 Dense layers
- Input: (30 frames, 126 normalized features)
- Parameters: 188,288
- Training: 5 independent runs, best by validation loss

Place it in the project root before running.

---

## Research

This project is accompanied by an IEEE-format research paper:

> *Real-Time Bidirectional Sign Language Translator: Sign-to-Speech, Speech-to-Sign, Networked Live Translation, and Web-Based Interface*  
---

## License

MIT License — see [LICENSE](LICENSE)

---

## Acknowledgements

- [ASL University (Lifeprint.com)](https://www.lifeprint.com) — Dr. Bill Vicars — ASL reference source
- [Gallaudet University ASL Connect](https://www.gallaudet.edu) — Fingerspelling reference
- [Google MediaPipe](https://mediapipe.dev) — Hand landmark detection
- [HuggingFace Transformers](https://huggingface.co) — T5-small
- [Meta Llama3 via Ollama](https://ollama.com) — Local LLM verification
