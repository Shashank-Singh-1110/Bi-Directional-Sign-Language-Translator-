import os
import re
import sys

import numpy as np

DATA_PATH = 'data/DATASET_NORM'

ACTIONS = [
    'Hello', 'Thanks', 'Yes', 'I LOVE YOU', 'No', 'Sorry',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rotate_vector(v, degrees):
    """Rotate both hands in the xy plane. z untouched."""
    rad = np.radians(degrees)
    c, s = np.cos(rad), np.sin(rad)

    out = v.copy()
    for base in (0, 63):
        pts = out[base:base + 63].reshape(21, 3)
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        pts[:, 0] = c * x - s * y
        pts[:, 1] = s * x + c * y

    return out


def load_frames(per_class=40):
    """Frames per class, sampled across sequences and timesteps."""
    frames = {}
    for action in ACTIONS:
        a_dir = os.path.join(DATA_PATH, action)
        if not os.path.isdir(a_dir):
            continue

        collected = []
        for seq in sorted(os.listdir(a_dir)):
            s_dir = os.path.join(a_dir, seq)
            if not os.path.isdir(s_dir):
                continue
            for f in (10, 14, 18, 22):
                fp = os.path.join(s_dir, f'{f}.npy')
                if os.path.exists(fp):
                    collected.append(np.load(fp))
            if len(collected) >= per_class:
                break

        if collected:
            frames[action] = np.array(collected[:per_class], dtype=np.float32)

    return frames


def parse_vector(path):
    with open(path) as fh:
        text = fh.read()
    nums = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', text)
    vals = [float(n) for n in nums]
    if len(vals) < 126:
        raise ValueError(f"Found only {len(vals)} numbers; expected >= 126")
    return np.array(vals[-126:], dtype=np.float32)


def baselines(frames, truth):
    """How similar are frames within a class, and across classes?"""
    own = frames[truth]

    intra = []
    for i in range(min(25, len(own))):
        for j in range(i + 1, min(25, len(own))):
            intra.append(cosine(own[i], own[j]))

    inter = []
    for action, arr in frames.items():
        if action == truth:
            continue
        for i in range(min(4, len(arr))):
            for j in range(min(4, len(own))):
                inter.append(cosine(own[j], arr[i]))

    return np.array(intra), np.array(inter)


def rank_against(vec, frames, truth):
    scores = {a: max(cosine(vec, f) for f in arr) for a, arr in frames.items()}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    rank = [a for a, _ in ranked].index(truth) + 1
    return ranked, rank, scores[truth]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    vec = parse_vector(sys.argv[1])
    truth = sys.argv[2]

    frames = load_frames()
    if truth not in frames:
        print(f"[ERROR] No data for '{truth}'")
        sys.exit(1)

    print("=" * 72)
    print("  What does a good match look like in this dataset?")
    print("=" * 72)

    intra, inter = baselines(frames, truth)

    print(f"\n  Same sign, different recordings ({truth}):")
    print(f"    mean {intra.mean():.3f}   median {np.median(intra):.3f}   "
          f"min {intra.min():.3f}   max {intra.max():.3f}")

    print(f"\n  {truth} vs every other sign:")
    print(f"    mean {inter.mean():.3f}   median {np.median(inter):.3f}   "
          f"max {inter.max():.3f}")

    print(f"\n  Separation: {intra.mean() - inter.mean():+.3f}")

    if intra.mean() - inter.mean() < 0.1:
        print("\n  WARNING: same-sign and different-sign similarities overlap.")
        print("  Cosine on a single frame is not very discriminative here,")
        print("  which limits what this diagnostic can conclude.")

    # ── Rotation sweep ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Rotation sweep")
    print("=" * 72)

    best = None
    rows = []

    for deg in range(0, 360, 5):
        rotated = rotate_vector(vec, deg)
        ranked, rank, score = rank_against(rotated, frames, truth)
        rows.append((deg, score, ranked[0][0], ranked[0][1], rank))
        if best is None or score > best[1]:
            best = (deg, score, ranked[0][0], rank)

    # Only print the interesting ones: top 12 by score
    rows.sort(key=lambda r: -r[1])
    print(f"\n{'angle':>7}{'sim to truth':>15}{'best match':>16}"
          f"{'its score':>12}{'rank':>7}")
    print("-" * 72)
    for deg, score, top, top_score, rank in rows[:12]:
        marker = '  <<<' if rank == 1 else ''
        print(f"{deg:>6}°{score:>15.3f}{top:>16}{top_score:>12.3f}"
              f"{rank:>7}{marker}")

    print("-" * 72)
    print(f"\n  Best angle: {best[0]}°  (similarity {best[1]:.3f}, "
          f"rank {best[3]})")

    # ── Full ranking at the best angle ──────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  Full ranking at {best[0]}°")
    print("=" * 72)

    ranked, rank, score = rank_against(rotate_vector(vec, best[0]),
                                       frames, truth)
    for i, (action, s) in enumerate(ranked[:10], 1):
        marker = '   <- performed' if action == truth else ''
        print(f"  {i:>2}. {action:<14}{s:.3f}{marker}")

    if rank > 10:
        print(f"  ...")
        print(f"  {rank:>2}. {truth:<14}{score:.3f}   <- performed")

    # ── Verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)

    threshold = np.percentile(intra, 10)

    if best[1] >= threshold and best[3] == 1:
        print(f"\n  Rotating by {best[0]}° brings the Android vector into the")
        print(f"  normal range for this sign ({best[1]:.3f} vs a same-sign")
        print(f"  10th percentile of {threshold:.3f}), and ranks it first.")
        print(f"\n  Apply that rotation in HandLandmarkerHelper.")
    elif best[1] < inter.mean() + 0.1:
        print(f"\n  Best similarity {best[1]:.3f} is no better than comparing")
        print(f"  this sign to unrelated signs ({inter.mean():.3f} mean).")
        print(f"\n  The Android landmarks are not a rotated, mirrored or")
        print(f"  reordered version of the training landmarks. The two")
        print(f"  MediaPipe models produce genuinely different geometry.")
        print(f"\n  The fix is to re-extract the dataset with MediaPipe")
        print(f"  HandLandmarker — the same model Android runs — and")
        print(f"  retrain. The recorded videos are not needed; the raw")
        print(f"  DATASET frames came from Holistic, so this means")
        print(f"  re-recording, or running HandLandmarker over saved video")
        print(f"  if any was kept.")
    else:
        print(f"\n  Partial match ({best[1]:.3f}, same-sign 10th percentile")
        print(f"  {threshold:.3f}). Something systematic still differs.")
        print(f"\n  Worth capturing three or four more vectors for different")
        print(f"  signs — one frame may have been caught mid-transition.")


if __name__ == '__main__':
    main()