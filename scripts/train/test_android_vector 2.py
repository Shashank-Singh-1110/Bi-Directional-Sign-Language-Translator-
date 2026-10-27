"""
test_android_vector.py — Does the retrained model handle real Android input?

The held-out evaluation cannot answer this: every sequence in the dataset is
~0.999 similar to every other, so the test set is as easy as the training
set and any model scores 100%.

The Android capture is the only genuinely out-of-distribution sample we
have. This runs it through a model and prints the ranking.

Usage:
    python scripts/train/test_android_vector.py android_vector.txt "I LOVE YOU"
    python scripts/train/test_android_vector.py android_vector.txt "I LOVE YOU" models/action_norm.h5
"""

import re
import sys

import numpy as np

SEQ_LEN = 30
DEFAULT_MODEL = 'models/action_aug.h5'
BASELINE_MODEL = 'models/action_norm.h5'

ACTIONS = [
    'Hello', 'Thanks', 'Yes', 'I LOVE YOU', 'No', 'Sorry',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]


def parse_vector(path):
    with open(path) as fh:
        nums = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', fh.read())
    vals = [float(n) for n in nums]
    if len(vals) < 126:
        raise ValueError(f"Found {len(vals)} numbers; expected >= 126")
    return np.array(vals[-126:], dtype=np.float32)


def predict(model_path, vec, truth, tf):
    model = tf.keras.models.load_model(model_path, compile=False)

    # The app feeds a held sign, so 30 near-identical frames is a fair
    # reconstruction of what the model sees in practice.
    seq = np.tile(vec, (SEQ_LEN, 1))[np.newaxis, ...].astype(np.float32)

    probs = model.predict(seq, verbose=0)[0]
    order = np.argsort(-probs)

    rank = int(np.where(order == ACTIONS.index(truth))[0][0]) + 1

    print(f"\n  {model_path}")
    print("  " + "-" * 44)
    for i in order[:6]:
        marker = '   <- performed' if ACTIONS[i] == truth else ''
        print(f"    {ACTIONS[i]:<14}{probs[i] * 100:>7.2f}%{marker}")

    if rank > 6:
        idx = ACTIONS.index(truth)
        print(f"    ...")
        print(f"    {truth:<14}{probs[idx] * 100:>7.2f}%   <- performed "
              f"(rank {rank})")

    return rank, float(probs[ACTIONS.index(truth)] * 100), float(probs.max() * 100)


def main():
    import os

    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    import tensorflow as tf

    vec_path, truth = sys.argv[1], sys.argv[2]
    vec = parse_vector(vec_path)

    print("=" * 56)
    print("  Android capture vs the trained models")
    print("=" * 56)
    print(f"  Performed: {truth}")

    results = {}

    for label, path in (('baseline', BASELINE_MODEL),
                        ('augmented', DEFAULT_MODEL)):
        if os.path.exists(path):
            results[label] = predict(path, vec, truth, tf)
        else:
            print(f"\n  {path} not found — skipped")

    print("\n" + "=" * 56)
    print("  VERDICT")
    print("=" * 56)

    for label, (rank, truth_conf, top_conf) in results.items():
        verdict = 'CORRECT' if rank == 1 else f'rank {rank}'
        print(f"  {label:<12} {verdict:<10} "
              f"{truth} at {truth_conf:.1f}%, top at {top_conf:.1f}%")

    if 'augmented' in results and 'baseline' in results:
        b_rank = results['baseline'][0]
        a_rank = results['augmented'][0]

        print()
        if a_rank == 1:
            print("  The augmented model gets it right on real Android input.")
            print("  Convert it and put it in the app.")
        elif a_rank < b_rank:
            print(f"  Improved ({b_rank} -> {a_rank}) but not correct yet.")
            print("  Augmentation is helping. Worth raising VARIANTS and the")
            print("  jitter sigma, since the gap is narrowing.")
        else:
            print(f"  No improvement ({b_rank} -> {a_rank}).")
            print("  Synthetic variation is not covering the real difference")
            print("  between MediaPipe Holistic and HandLandmarker. Recording")
            print("  new data with HandLandmarker is the remaining option.")


if __name__ == '__main__':
    main()