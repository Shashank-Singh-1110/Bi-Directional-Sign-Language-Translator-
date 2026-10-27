# Demo Guide

Everything you need to run SignBridge for a live demonstration.

---

## Prerequisites

- Python 3.12
- MacBook (tested on M1/M5) or any modern laptop with webcam
- A microphone
- `action_norm.h5` model file — download from [Releases](../../releases)

---

## Setup (one time)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/signbridge-asl-translator.git
cd signbridge-asl-translator

# 2. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Apple Silicon only — fix native dependencies
brew install flac portaudio
pip install "setuptools==67.8.0" --force-reinstall

# 5. Place model file
# Download action_norm.h5 from Releases and put it in the project root

# 6. (Optional) Install Ollama for LLM verification
brew install ollama
ollama pull llama3
```

---

## Running the Web App

```bash
# Terminal 1 — (optional) start Ollama for RAG verification
ollama run llama3

# Terminal 2 — start the server
source .venv/bin/activate
python app.py
```

Open `http://localhost:5001` in your browser.

---

## Tab-by-Tab Demo Flow

### ✋ Sign to Speech

1. Click **▶ Start Camera** — your webcam feed appears
2. Sign any ASL letter or gesture — chips appear in the Sign Buffer
3. Click **⚡ Convert to Sentence** — T5 converts to grammatical English
4. Click **🔊 Speak** — TTS reads the sentence aloud
5. The **ASL Verification** panel shows automatically after each detection

**Suggested demo signs:** Hello → I LOVE YOU → Y-O-U → Convert

### 🎤 Speech to Sign

1. Click the microphone button — it turns red when listening
2. Say a sentence clearly: *"Hello how are you"*
3. GIFs play in sequence for each word
4. Or type directly in the text box and click **Translate to Sign**

### 🌐 Live Network

**On your machine:**
1. Go to Live Network tab
2. Enter your name in "Create Session"
3. Click 🎲 GEN to generate a room code
4. Click **📡 Create Room**

**On peer's machine** (any device on same WiFi):
1. Open `http://YOUR_IP:5001`
2. Go to Live Network tab
3. Enter name, enter the room code, click **🔗 Join Room**

**Get your IP:**
```bash
ipconfig getifaddr en0    # Mac WiFi
```

Once connected:
- You sign → peer's screen shows GIFs + TTS speaks the sentence
- Peer speaks → your screen shows GIFs

---

## Running Standalone Scripts

### Sign detection only (OpenCV window)
```bash
python Translator.py
```

### Speech to Sign only
```bash
python Speech_TO_Sign.py
```

### Peer-to-peer TCP mode (no browser)
```bash
# Machine 1 (host)
python live_translator.py --mode host

# Machine 2 (client)
python live_translator.py --mode client --host 192.168.1.X
```

### Collect new training data
```bash
python data_collection.py
# Press SPACE to record each sequence
# Press Q to quit
```

### Train/retrain the model
```bash
python normalize_and_retrain.py
# Runs 5 training sessions, saves best model as action_norm.h5
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: distutils` | `pip install setuptools==67.8.0 --force-reinstall` |
| `write() before start_response` | Make sure `import eventlet; eventlet.monkey_patch()` are first 2 lines of app.py |
| Camera not showing | Check browser permissions — allow camera for localhost |
| Speech not working | Allow microphone in browser. Check you're on `http://` not `file://` |
| Ollama verification not working | Run `ollama run llama3` in a separate terminal before starting app.py |
| PyAudio install fails | `brew install portaudio` first |
| FLAC error on speech | `brew install flac` |

---

## Demo Tips

- **Camera distance:** 40–80cm works best. The normalization handles variation but extreme distances reduce landmark quality.
- **Lighting:** Even, front-facing light. Avoid backlight (window behind you).
- **Sign speed:** Hold each sign for about 1.5 seconds. The stability buffer needs 10 consecutive agreeing frames.
- **Sign J and Z:** These are motion signs — trace the letter path slowly and clearly.
- **Live Network:** Use a WiFi hotspot if a shared network isn't available. Both devices just need to reach the same IP.
