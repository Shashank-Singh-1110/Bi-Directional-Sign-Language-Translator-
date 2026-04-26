import cv2
import numpy as np
import mediapipe as mp
import time
import os
import collections
from tensorflow.keras.models import load_model
from Gloss import convert_and_speak, group_letters

ACTIONS = np.array([
    'Hello', 'Thanks', 'Yes', 'I LOVE YOU', 'No', 'Sorry',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
])

MODEL_PATH       = 'action_norm.h5'
SEQUENCE_LENGTH  = 30
STABILITY_FRAMES = 10
COOLDOWN_SECONDS = 3.0
MAX_BUFFER_LEN   = 20
FEATURE_DIM      = 126

GESTURE_SIGNS  = {'Hello', 'Thanks', 'Yes', 'I LOVE YOU', 'No', 'Sorry'}
GESTURE_THRESH = 0.70
LETTER_THRESH  = 0.88
MIN_CONF_GAP   = 0.25

MOTION_GATE_FRAMES = 6
MOTION_THRESHOLD   = 0.012

mp_holistic   = mp.solutions.holistic
mp_drawing    = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles


def mediapipe_detection(frame, model):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = model.process(rgb)
    rgb.flags.writeable = True
    return results


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


def draw_landmarks(frame, results):
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_draw_style.get_default_hand_landmarks_style(),
            mp_draw_style.get_default_hand_connections_style())
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_draw_style.get_default_hand_landmarks_style(),
            mp_draw_style.get_default_hand_connections_style())


def is_hand_still(sequence):
    if len(sequence) < MOTION_GATE_FRAMES:
        return False
    arr = np.array(list(sequence)[-MOTION_GATE_FRAMES:])
    return np.std(arr, axis=0).mean() < MOTION_THRESHOLD


def draw_ui(frame, word, confidence, sign_buffer, last_sentence,
            buffer_len, is_paused, top_preds, hand_still, gap_ok):
    h, w = frame.shape[:2]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 95), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if is_paused:
        cv2.putText(frame, "PAUSED — SPACE to resume",
                    (w // 2 - 160, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 165, 255), 2)
    elif word:
        thresh     = GESTURE_THRESH if word in GESTURE_SIGNS else LETTER_THRESH
        conf_color = (0, 200, 100) if confidence >= thresh else (0, 165, 255)

        cv2.putText(frame, word, (20, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame, f"{confidence * 100:.0f}%",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)

        motion_col = (0, 255, 0) if hand_still else (0, 0, 255)
        cv2.putText(frame, "STILL" if hand_still else "MOVING",
                    (w - 130, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, motion_col, 2)
        gap_col = (0, 255, 0) if gap_ok else (0, 165, 255)
        cv2.putText(frame, "GAP OK" if gap_ok else "LOW GAP",
                    (w - 130, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gap_col, 1)

        bx, by, bw, bh = 120, 58, 220, 16
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
        cv2.rectangle(frame, (bx, by),
                      (bx + int(bw * min(confidence, 1.0)), by + bh),
                      conf_color, -1)

    for i, (label, prob) in enumerate(top_preds[:3]):
        alpha = max(0.4, 1.0 - i * 0.25)
        col   = (int(200 * alpha),) * 3
        cv2.putText(frame, f"{label}: {prob * 100:.0f}%",
                    (w - 190, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

    # Sequence progress bar
    bar_w  = w - 40
    filled = int(bar_w * buffer_len / SEQUENCE_LENGTH)
    cv2.rectangle(frame, (20, h - 10), (20 + bar_w, h - 3), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, h - 10), (20 + filled, h - 3), (80, 80, 180), -1)

    # Bottom area — two rows
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 100), (w, h - 12), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)

    # Row 1 — raw sign buffer (what you've signed)
    gloss_preview = ' '.join(sign_buffer) if sign_buffer else "Sign something..."
    # Show grouped preview
    if sign_buffer:
        grouped = group_letters(sign_buffer)
        gloss_preview = ' · '.join(grouped)
    cv2.putText(frame, gloss_preview[:60],  # truncate if too long
                (20, h - 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 255), 2)

    # Row 2 — last T5 sentence output
    if last_sentence:
        cv2.putText(frame, f'"{last_sentence}"',
                    (20, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 120), 2)

    # Controls hint
    cv2.putText(frame, "Q:quit  C:clear  SPACE:pause  ENTER:convert",
                (20, h - 115), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)

    return frame


def main():
    print("=" * 55)
    print("  SIGN LANGUAGE TRANSLATOR + T5 GLOSS")
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Features: {FEATURE_DIM}-dim (hand only, normalized)")
    print(f"  Classes : {len(ACTIONS)}")
    print("  ENTER   : convert buffer to sentence via T5")
    print("  C       : clear buffer")
    print("=" * 55)

    model = load_model(MODEL_PATH)
    print("[OK] Model loaded")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    sequence             = collections.deque(maxlen=SEQUENCE_LENGTH)
    stability_buffer     = collections.deque(maxlen=STABILITY_FRAMES)
    sign_buffer          = []     # raw detected signs e.g. ['H','E','L','L','O']
    last_sentence        = ""     # last T5 output
    last_prediction_time = 0
    current_word         = ""
    current_conf         = 0.0
    top_preds            = []
    is_paused            = False
    hand_still           = False
    gap_ok               = False

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            results = mediapipe_detection(frame, holistic)
            draw_landmarks(frame, results)

            if not is_paused:
                keypoints  = extract_keypoints(results)
                sequence.append(keypoints)
                hand_still = is_hand_still(sequence)

                if len(sequence) == SEQUENCE_LENGTH:
                    X          = np.expand_dims(np.array(sequence), axis=0)
                    prediction = model.predict(X, verbose=0)[0]

                    top3_idx  = np.argsort(prediction)[-3:][::-1]
                    top_preds = [(ACTIONS[i], prediction[i]) for i in top3_idx]

                    best_word = top_preds[0][0]
                    best_conf = float(top_preds[0][1])
                    sec_conf  = float(top_preds[1][1])
                    gap_ok    = (best_conf - sec_conf) >= MIN_CONF_GAP

                    current_word = best_word
                    current_conf = best_conf

                    hand_visible = (results.left_hand_landmarks is not None or
                                    results.right_hand_landmarks is not None)
                    threshold    = GESTURE_THRESH if best_word in GESTURE_SIGNS \
                                   else LETTER_THRESH

                    if (hand_visible and best_conf >= threshold
                            and gap_ok and hand_still):
                        stability_buffer.append(best_word)
                    else:
                        stability_buffer.append("")

                    stable = (
                        len(stability_buffer) == STABILITY_FRAMES and
                        len(set(stability_buffer)) == 1 and
                        stability_buffer[0] != ""
                    )

                    now = time.time()
                    if stable and (now - last_prediction_time) > COOLDOWN_SECONDS:
                        word = stability_buffer[0]
                        # Add to sign buffer (allow repeats for spelling)
                        if len(sign_buffer) < MAX_BUFFER_LEN:
                            sign_buffer.append(word)
                            print(f"  + '{word}' ({best_conf*100:.0f}%)  "
                                  f"buffer: {sign_buffer}")
                        last_prediction_time = now
                        stability_buffer.clear()

            frame = draw_ui(frame, current_word, current_conf,
                            sign_buffer, last_sentence,
                            len(sequence), is_paused,
                            top_preds, hand_still, gap_ok)
            cv2.imshow('Sign Language Translator', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == 13:   # ENTER — convert buffer to sentence
                if sign_buffer:
                    print(f"\n  [ENTER] Converting: {sign_buffer}")
                    _, sentence, method = convert_and_speak(sign_buffer,
                                                            verbose=True)
                    last_sentence = sentence
                    sign_buffer.clear()
                    stability_buffer.clear()
                    print(f"  [T5] '{sentence}' ({method})\n")
                else:
                    print("  [ENTER] Buffer empty — sign something first")

            elif key == ord('c'):
                sign_buffer.clear()
                stability_buffer.clear()
                last_sentence = ""
                current_word  = ""
                print("  Cleared")

            elif key == ord(' '):
                is_paused = not is_paused
                sequence.clear()
                stability_buffer.clear()
                print(f"  {'Paused' if is_paused else 'Resumed'}")

            elif key == ord('z'):   # undo last sign
                if sign_buffer:
                    removed = sign_buffer.pop()
                    print(f"  Removed '{removed}'  buffer: {sign_buffer}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == '__main__':
    main()