import os
import json

import numpy as np

RAW_DATA  = 'DATASET'          # raw 258-dim frames
OUT_FILE  = '../../android_fixtures.json'
N_CASES   = 12                 # spread across classes and hand configurations

ACTIONS = [
    'Hello', 'Thanks', 'Yes', 'I LOVE YOU', 'No', 'Sorry',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]


# ── The reference implementation, copied verbatim from
#    normalize_and_retrain.py so fixtures cannot drift ──────────────────
def normalize_hand(hand_63):
    pts = hand_63.reshape(21, 3)
    wrist = pts[0].copy()
    pts = pts - wrist

    dists = np.linalg.norm(pts, axis=1)
    max_d = dists.max()
    if max_d > 1e-6:
        pts = pts / max_d

    return pts.flatten()


def normalize_keypoints(raw_258):
    lh_raw = raw_258[132:195]
    rh_raw = raw_258[195:258]
    return np.concatenate([normalize_hand(lh_raw), normalize_hand(rh_raw)])


def collect_frames():
    """Pick frames covering both-hands, left-only, right-only and no-hands."""
    buckets = {'both': [], 'left_only': [], 'right_only': [], 'none': []}

    for action in ACTIONS:
        a_dir = os.path.join(RAW_DATA, action)
        if not os.path.isdir(a_dir):
            continue

        for seq in sorted(os.listdir(a_dir))[:3]:
            s_dir = os.path.join(a_dir, seq)
            if not os.path.isdir(s_dir):
                continue

            for f in (0, 10, 20, 29):
                fp = os.path.join(s_dir, f'{f}.npy')
                if not os.path.exists(fp):
                    continue

                raw = np.load(fp)
                if raw.shape[0] != 258:
                    continue

                has_l = bool(np.any(raw[132:195] != 0))
                has_r = bool(np.any(raw[195:258] != 0))

                if has_l and has_r:
                    key = 'both'
                elif has_l:
                    key = 'left_only'
                elif has_r:
                    key = 'right_only'
                else:
                    key = 'none'

                if len(buckets[key]) < N_CASES // 4 + 1:
                    buckets[key].append((action, seq, f, raw))

    return buckets


def main():
    if not os.path.isdir(RAW_DATA):
        print(f"[ERROR] '{RAW_DATA}' not found.")
        print("        Fixtures must come from raw 258-dim frames, not")
        print("        DATASET_NORM (which is already normalized).")
        return

    buckets = collect_frames()
    cases = []

    for kind, items in buckets.items():
        for action, seq, frame, raw in items:
            norm = normalize_keypoints(raw)
            cases.append({
                'name':   f'{action}_seq{seq}_f{frame}_{kind}',
                'kind':   kind,
                'raw258': [round(float(v), 8) for v in raw],
                'expected126': [round(float(v), 8) for v in norm],
            })

    with open(OUT_FILE, 'w') as fh:
        json.dump({'cases': cases}, fh, indent=1)

    print(f"[DONE] Wrote {len(cases)} fixtures -> {OUT_FILE}")
    for kind in buckets:
        n = sum(1 for c in cases if c['kind'] == kind)
        print(f"       {kind:<12} {n}")

    if not any(c['kind'] == 'none' for c in cases):
        print("\n[WARN] No empty-hand frames found. The zero-input path")
        print("       (max_d <= 1e-6) will go untested — add a synthetic case.")

    print(f"\n[NEXT] Copy {OUT_FILE} to:")
    print("       app/src/test/resources/android_fixtures.json")


if __name__ == '__main__':
    main()