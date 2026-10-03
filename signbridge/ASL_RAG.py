import json
import os
import re
import time
import requests
from difflib import SequenceMatcher

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
KB_PATH      = os.path.join(os.path.dirname(__file__), "asl_knowledge_base.json")

_kb = []
_kb_index = {}


def _load_kb():
    global _kb, _kb_index
    if _kb:
        return
    with open(KB_PATH, 'r') as f:
        _kb = json.load(f)
    for entry in _kb:
        _kb_index[entry['sign'].upper()] = entry
    print(f"[RAG] Knowledge base loaded: {len(_kb)} ASL signs")


_load_kb()


def retrieve(sign: str) -> dict | None:
    key = sign.upper().strip()
    if key in _kb_index:
        return _kb_index[key]
    aliases = {
        'I LOVE YOU': 'I LOVE YOU',
        'ILY': 'I LOVE YOU',
        'THANKYOU': 'Thanks',
        'THANK YOU': 'Thanks',
    }
    if key in aliases:
        return _kb_index.get(aliases[key].upper())
    best, best_score = None, 0
    for k, entry in _kb_index.items():
        score = SequenceMatcher(None, key, k).ratio()
        if score > best_score:
            best_score = score
            best = entry

    return best if best_score > 0.6 else None

def _call_llama(prompt: str, timeout: int = 10) -> str:
    key = prompt.upper().strip()
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature":0.1,
                "num_predict":120,
                "top_p":0.9,
            }
        },timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    except requests.exceptions.ConnectionError:
        return "OLLAMA_OFFLINE"
    except requests.exceptions.Timeout:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"


def verify_sign(sign: str, confidence: float = 0.0) -> dict:
    t0 = time.time()
    entry = retrieve(sign)

    if not entry:
        return {
            'sign': sign,
            'verified': False,
            'confidence': confidence,
            'status': 'not_found',
            'description': 'Sign not found in ASL knowledge base.',
            'handshape': '—',
            'source': '—',
            'llm_verdict': 'No reference data available for this sign.',
            'latency_ms': int((time.time() - t0) * 1000),
        }

    conf_pct = round(confidence * 100, 1)

    prompt = f"""You are an ASL (American Sign Language) verification assistant.

A sign language recognition system detected the sign "{sign}" with {conf_pct}% confidence.

Here is the authentic ASL reference for "{entry['sign']}":
- Description: {entry['description']}
- Handshape: {entry['handshape']}
- Movement: {entry['movement']}
- Location: {entry['location']}
- Source: {entry['source']}

Based on this authentic reference, provide a ONE sentence verdict on whether this detection is credible.
Start your response with either VERIFIED or UNCERTAIN.
Be concise. Max 30 words."""

    llm_response = _call_llama(prompt)
    if llm_response in ("OLLAMA_OFFLINE", "TIMEOUT") or llm_response.startswith("ERROR"):
        verified = confidence >= 0.75
        status = 'verified' if verified else 'low_confidence'
        verdict = f"Verified against ASL reference (confidence: {conf_pct}%)" if verified else f"Low model confidence: {conf_pct}%"
    else:
        upper = llm_response.upper()
        if 'VERIFIED' in upper[:20]:
            verified = True
            status = 'verified'
        elif confidence >= 0.92:
            verified = True
            status = 'verified'
        else:
            verified = False
            status = 'low_confidence'
        verdict = llm_response[:120]

    latency = int((time.time() - t0) * 1000)

    return {
        'sign': entry['sign'],
        'verified': verified,
        'confidence': confidence,
        'status': status,
        'description': entry['description'],
        'handshape': entry['handshape'],
        'movement': entry.get('movement', '—'),
        'location': entry.get('location', '—'),
        'source': entry['source'],
        'llm_verdict': verdict,
        'latency_ms': latency,
    }

def verify_buffer(buffer: list) -> list:
    return [verify_sign(sign) for sign in buffer]

if __name__ == '__main__':
    tests = ['Hello', 'Thanks', 'A', 'J', 'I LOVE YOU', 'Z']
    for sign in tests:
        r = verify_sign(sign, confidence=0.95)
        icon = '✓' if r['verified'] else '✗'
        print(f"[{icon}] {r['sign']:12} | {r['status']:15} | {r['latency_ms']}ms | {r['llm_verdict'][:60]}")