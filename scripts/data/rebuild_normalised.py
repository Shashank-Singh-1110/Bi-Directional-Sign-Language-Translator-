import os
import sys

import numpy as np

RAW_PATH  = 'data/DATASET'
NORM_PATH = 'data/DATASET_NORM'
SEQ_LEN   = 30

LEFT_SLICE  = slice(132, 195)
RIGHT_SLICE = slice(195, 258)
EPSILON     = 1e-6


def normalize_hand(hand_63):
    """Wrist to origin, then scale by the furthest landmark distance."""
    pts = hand_63.reshape(21, 3)
    pts = pts - pts[0]

    max_d = np.linalg.norm(pts, axis=1).max()
    if max_d > EPSILON:
        pts = pts / max_d

    return pts.flatten()


def normalize_keypoints(raw_258):
    """258-dim Holistic vector -> 126-dim hands-only normalized vector."""
    return np.concatenate([
        normalize_hand(raw_258[LEFT_SLICE]),
        normalize_hand(raw_258[RIGHT_SLICE]),
    ])


def main():
    if not os.path.isdir(RAW_PATH):
        print(f"[ERROR] '{RAW_PATH}' not found.")
        print("        The raw dataset is what everything else is derived")
        print("        from — without it, nothing here can be rebuilt.")
        sys.exit(1)

    actions = sorted(
        d for d in os.listdir(RAW_PATH)
        if os.path.isdir(os.path.join(RAW_PATH, d))
    )

    print(f"[NORM] {RAW_PATH} -> {NORM_PATH}")
    print(f"[NORM] {len(actions)} classes")

    total = 0
    skipped = 0

    for action in actions:
        src_action = os.path.join(RAW_PATH, action)
        dst_action = os.path.join(NORM_PATH, action)

        for seq in sorted(os.listdir(src_action)):
            src_seq = os.path.join(src_action, seq)
            if not os.path.isdir(src_seq):
                continue

            dst_seq = os.path.join(dst_action, seq)
            os.makedirs(dst_seq, exist_ok=True)

            for f in range(SEQ_LEN):
                src = os.path.join(src_seq, f'{f}.npy')
                if not os.path.exists(src):
                    continue

                raw = np.load(src)
                if raw.shape[0] != 258:
                    skipped += 1
                    continue

                np.save(os.path.join(dst_seq, f'{f}.npy'),
                        normalize_keypoints(raw))
                total += 1

        print(f"  {action}")

    print(f"\n[DONE] {total} frames normalized to 126 dimensions")
    if skipped:
        print(f"[WARN] {skipped} frames skipped — unexpected shape")

    # Confirm the output is loadable in the shape the model expects
    seqs = sum(
        1
        for a in os.listdir(NORM_PATH)
        if os.path.isdir(os.path.join(NORM_PATH, a))
        for s in os.listdir(os.path.join(NORM_PATH, a))
        if os.path.isdir(os.path.join(NORM_PATH, a, s))
    )
    print(f"[DONE] {seqs} sequences written")


if __name__ == '__main__':
    main()