import json
import os
import re
import time

import requests

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434/api/generate')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3')
TIMEOUT = 30

KB_PATH = os.path.join(os.path.dirname(__file__), 'asl_knowledge_base.json')

MARKER = re.compile(r'\[(SHOW|PRACTICE):([^\]]+)\]', re.IGNORECASE)

_kb = None


def knowledge_base():
    global _kb
    if _kb is None:
        with open(KB_PATH) as fh:
            _kb = json.load(fh)
        print(f"[TUTOR] Knowledge base loaded: {len(_kb)} signs")
    return _kb


def lookup(name):
    """Find a sign by name, case-insensitively."""
    if not name:
        return None
    target = name.strip().lower()
    for entry in knowledge_base():
        if entry.get('sign', '').lower() == target:
            return entry
    return None


def available_signs():
    return [e.get('sign', '') for e in knowledge_base()]


SYSTEM_PROMPT = """You are a patient ASL tutor inside SignBridge, an app that
recognises American Sign Language through a webcam.

You teach hearing learners how to sign, and how to communicate respectfully
with Deaf and hard-of-hearing people.

TOOLS — put these on their own line to act:

  [SHOW:<sign>]      Plays an animation of the sign. Use it whenever you
                     describe how a sign is made.
  [PRACTICE:<sign>]  Turns on the learner's camera so they can try it. Use
                     it after demonstrating, when they are ready to attempt.

Only use signs from this list: {signs}

RULES
- Describe handshape, movement and location in plain words before showing.
- One sign at a time. Do not overload.
- After [PRACTICE:...] stop and wait. Do not assume they succeeded.
- If asked about something outside ASL, answer briefly and steer back.
- Never use "deaf and dumb" or "mute". Say Deaf, hard of hearing, or
  non-speaking.
- Keep replies short. Two or three sentences, then a tool call.

{context}"""


def build_context(sign_entry):
    if not sign_entry:
        return ""

    return (
        "\nREFERENCE for the sign currently being taught:\n"
        f"  Sign      : {sign_entry.get('sign')}\n"
        f"  Handshape : {sign_entry.get('handshape')}\n"
        f"  Movement  : {sign_entry.get('movement')}\n"
        f"  Location  : {sign_entry.get('location')}\n"
        f"  Notes     : {sign_entry.get('description')}\n"
        f"  Source    : {sign_entry.get('source')}\n"
        "Use these exact details. Do not invent handshapes."
    )


def call_ollama(prompt, system):
    """Returns (text, latency_ms). Raises on failure."""
    start = time.time()

    resp = requests.post(
        OLLAMA_URL,
        json={
            'model': OLLAMA_MODEL,
            'prompt': prompt,
            'system': system,
            'stream': False,
            'options': {'temperature': 0.7, 'num_predict': 300},
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()

    text = resp.json().get('response', '').strip()
    return text, int((time.time() - start) * 1000)


def parse_markers(text):
    """Strip markers out of the reply and return the actions they requested."""
    actions = []

    for kind, name in MARKER.findall(text):
        entry = lookup(name)
        if entry:
            actions.append({
                'type': kind.lower(),
                'sign': entry.get('sign'),
                'handshape': entry.get('handshape'),
                'movement': entry.get('movement'),
                'location': entry.get('location'),
            })

    clean = MARKER.sub('', text).strip()
    clean = re.sub(r'\n{3,}', '\n\n', clean)

    return clean, actions


def format_history(history, limit=6):
    lines = []
    for turn in history[-limit:]:
        role = 'Learner' if turn.get('role') == 'user' else 'Tutor'
        lines.append(f"{role}: {turn.get('content', '')}")
    return '\n'.join(lines)


# ── Public API ─────────────────────────────────────────────────────────
def chat(message, history=None, current_sign=None):
    """
    One turn of conversation.

    Returns a dict with the reply text, any actions the LLM requested, and
    which engine produced it.
    """
    history = history or []
    entry = lookup(current_sign) if current_sign else None

    system = SYSTEM_PROMPT.format(
        signs=', '.join(available_signs()),
        context=build_context(entry),
    )

    prompt = message
    if history:
        prompt = f"{format_history(history)}\nLearner: {message}\nTutor:"

    try:
        raw, latency = call_ollama(prompt, system)
        text, actions = parse_markers(raw)

        return {
            'reply': text or "Which sign would you like to learn?",
            'actions': actions,
            'engine': OLLAMA_MODEL,
            'latency_ms': latency,
        }

    except Exception as e:
        print(f"[TUTOR] Ollama unavailable: {e}")
        return fallback(message)


def fallback(message):
    """
    Knowledge-base-only reply when Ollama is unreachable.

    Matches a sign name in the message and reads out its entry. Not a
    conversation, but it keeps the tab usable offline.
    """
    lowered = message.lower()

    for entry in knowledge_base():
        name = entry.get('sign', '')
        if name and name.lower() in lowered:
            return {
                'reply': (
                    f"**{name}** — {entry.get('description')}\n\n"
                    f"Handshape: {entry.get('handshape')}\n"
                    f"Movement: {entry.get('movement')}\n"
                    f"Location: {entry.get('location')}"
                ),
                'actions': [{
                    'type': 'show',
                    'sign': name,
                    'handshape': entry.get('handshape'),
                    'movement': entry.get('movement'),
                    'location': entry.get('location'),
                }],
                'engine': 'knowledge base',
                'latency_ms': 0,
            }

    return {
        'reply': (
            "The language model is not running, so I can only look signs up "
            "directly. Name one and I will show it — for example, "
            "\"show me Thanks\".\n\n"
            f"Available: {', '.join(available_signs()[:12])}…"
        ),
        'actions': [],
        'engine': 'knowledge base',
        'latency_ms': 0,
    }


def grade(expected, detected, confidence):
    """
    Feedback on a practice attempt.

    The LLM is given what the sign should look like and what the vision
    model saw, so the feedback is specific rather than a bare right/wrong.
    """
    entry = lookup(expected)
    correct = (detected or '').lower() == (expected or '').lower()

    if not entry:
        return {
            'reply': f"I don't have reference details for {expected}.",
            'correct': correct,
            'actions': [],
            'engine': 'none',
            'latency_ms': 0,
        }

    system = (
        "You are an ASL tutor giving feedback on one practice attempt. "
        "Be encouraging and specific. Two or three sentences. "
        "If they were wrong, name the likely mistake using the reference "
        "details, and offer [SHOW:<sign>] to demonstrate again."
    )

    prompt = (
        f"The learner was asked to sign '{expected}'.\n"
        f"Reference: handshape {entry.get('handshape')}, "
        f"movement {entry.get('movement')}, "
        f"location {entry.get('location')}.\n\n"
    )

    if correct:
        prompt += (
            f"The recognition model identified it correctly at "
            f"{confidence:.0f}% confidence. Confirm and say what they did "
            f"well, referring to the handshape."
        )
    else:
        wrong = lookup(detected)
        prompt += (
            f"The model saw '{detected}' instead "
            f"({confidence:.0f}% confidence).\n"
        )
        if wrong:
            prompt += (
                f"'{detected}' uses handshape {wrong.get('handshape')} and "
                f"movement {wrong.get('movement')}.\n"
            )
        prompt += "Explain the difference and how to correct it."

    try:
        raw, latency = call_ollama(prompt, system)
        text, actions = parse_markers(raw)

        return {
            'reply': text,
            'correct': correct,
            'actions': actions,
            'engine': OLLAMA_MODEL,
            'latency_ms': latency,
        }

    except Exception as e:
        print(f"[TUTOR] Grading fell back: {e}")

        if correct:
            text = (
                f"That's **{expected}** — recognised at "
                f"{confidence:.0f}% confidence. Your handshape "
                f"({entry.get('handshape')}) came through clearly."
            )
        else:
            text = (
                f"That read as **{detected}** rather than **{expected}**. "
                f"For {expected}, the handshape is "
                f"{entry.get('handshape')} and the movement is "
                f"{entry.get('movement')}. Try again."
            )

        return {
            'reply': text,
            'correct': correct,
            'actions': [] if correct else [{'type': 'show', 'sign': expected}],
            'engine': 'knowledge base',
            'latency_ms': 0,
        }


def health():
    """Is the LLM reachable?"""
    try:
        base = OLLAMA_URL.rsplit('/api/', 1)[0]
        resp = requests.get(f"{base}/api/tags", timeout=3)
        models = [m['name'] for m in resp.json().get('models', [])]
        return {
            'available': True,
            'model': OLLAMA_MODEL,
            'installed': models,
            'signs': len(knowledge_base()),
        }
    except Exception:
        return {
            'available': False,
            'model': None,
            'installed': [],
            'signs': len(knowledge_base()),
        }