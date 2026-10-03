import re
import threading
import pyttsx3

_tokenizer = None
_t5_model = None


def _load_t5():
    global _tokenizer, _t5_model
    if _t5_model is None:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        import torch
        print("[T5] Loading T5-small...")
        _tokenizer = T5Tokenizer.from_pretrained("t5-small")
        _t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        _t5_model.eval()
        print("[T5] Ready ✓")


LETTERS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

GESTURE_SIGNS = {'HELLO', 'THANKS', 'YES', 'I LOVE YOU', 'NO', 'SORRY'}

RULES = {
    ('HELLO',):                         "Hello!",
    ('SORRY',):                         "I'm sorry.",
    ('THANKS',):                        "Thank you.",
    ('YES',):                           "Yes.",
    ('NO',):                            "No.",
    ('I LOVE YOU',):                    "I love you!",
    ('HELLO', 'THANKS'):                "Hello, thank you!",
    ('HELLO', 'YES'):                   "Hello, yes.",
    ('HELLO', 'SORRY'):                 "Hello, I'm sorry.",
    ('HELLO', 'I LOVE YOU'):            "Hello, I love you!",
    ('NO', 'SORRY'):                    "No, I'm sorry.",
    ('YES', 'THANKS'):                  "Yes, thank you.",
    ('NO', 'THANKS'):                   "No, thank you.",
    ('SORRY', 'YES'):                   "Yes, I'm sorry.",
}


def group_letters(sign_buffer):
    """
    Convert raw sign buffer into gloss tokens.
    Consecutive single letters are joined into words.

    e.g. ['H','E','L','L','O','I LOVE YOU','Y','O','U']
         → ['HELLO', 'I LOVE YOU', 'YOU']
    """
    gloss = []
    letters = []

    for sign in sign_buffer:
        sign_up = sign.upper()
        if sign_up in LETTERS:
            letters.append(sign_up)
        else:
            # Flush accumulated letters as a word
            if letters:
                gloss.append(''.join(letters))
                letters = []
            gloss.append(sign_up)

    if letters:
        gloss.append(''.join(letters))

    return gloss

def rule_lookup(gloss_tokens):
    key = tuple(gloss_tokens)
    return RULES.get(key, None)


def t5_convert(gloss_tokens, max_length=64):
    _load_t5()
    import torch

    gloss_str = ' '.join(gloss_tokens)
    prompt = f"convert ASL gloss to English sentence: {gloss_str}"

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        max_length=128,
        truncation=True
    )

    with torch.no_grad():
        outputs = _t5_model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

    result = _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return result


def post_process(text):
    if not text:
        return text
    text = text.strip()
    text = text[0].upper() + text[1:]
    if text[-1] not in '.!?':
        text += '.'
    text = re.sub(r'\s+', ' ', text)
    return text

def speak(text):
    def _run():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        engine.say(text)
        engine.runAndWait()

    threading.Thread(target=_run, daemon=True).start()


def convert_and_speak(sign_buffer, verbose=True):

    if not sign_buffer:
        return [], "", "empty"

    # Step 1 — group letters
    gloss = group_letters(sign_buffer)
    if verbose:
        print(f"  [GLOSS] Raw buffer : {sign_buffer}")
        print(f"  [GLOSS] Grouped    : {gloss}")

    # Step 2 — rule lookup
    sentence = rule_lookup(gloss)
    method = "rule"

    if sentence is None:
        # Step 3 — T5
        try:
            raw = t5_convert(gloss)
            sentence = post_process(raw)
            method = "t5"
        except Exception as e:
            print(f"  [T5] Error: {e}")
            # Fallback — join gloss tokens
            sentence = post_process(' '.join(t.lower() for t in gloss))
            method = "fallback"

    if verbose:
        print(f"  [GLOSS] Sentence   : '{sentence}' ({method})")

    # Step 4 — speak
    speak(sentence)

    return gloss, sentence, method


# ── Test ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_cases = [
        # Letter spelling
        ['H', 'E', 'L', 'L', 'O'],
        ['Y', 'O', 'U'],
        ['H', 'O', 'W', 'A', 'R', 'E', 'Y', 'O', 'U'],

        # Gesture only
        ['Hello'],
        ['I LOVE YOU'],
        ['Hello', 'Thanks'],

        # Mixed
        ['Hello', 'I LOVE YOU', 'Y', 'O', 'U'],
        ['S', 'O', 'R', 'R', 'Y', 'No'],
        ['Yes', 'I', 'A', 'M', 'O', 'K'],
        ['Hello', 'M', 'Y', 'N', 'A', 'M', 'E', 'I', 'S', 'J', 'O', 'H', 'N'],
        ['Thanks', 'Y', 'O', 'U'],
    ]

    print("=" * 55)
    print("  GLOSS → ENGLISH TEST")
    print("=" * 55)
    for buf in test_cases:
        gloss, sentence, method = convert_and_speak(buf, verbose=False)
        print(f"  {buf}")
        print(f"  → gloss: {gloss}")
        print(f"  → '{sentence}' [{method}]")
        print()
