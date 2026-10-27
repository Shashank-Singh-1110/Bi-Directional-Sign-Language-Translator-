"""
scripts/data/tune_detection.py — Find the preprocessing that detects best

Surrey's images are tight crops: roughly 110x120 px with the hand touching
every edge. MediaPipe's palm detector expects a hand occupying part of a
larger scene, so on these it fails about a third of the time, and unevenly
across letters — which would distort the class balance of any model trained
on what survives.

Padding and upscaling are affine and uniform, so wrist-origin normalisation
divides them straight back out: the landmarks move, the 126-dim feature
vector does not. That makes them free to apply.

This tries each variant over the same strided sample and reports the
detection rate, so the extractor can be set from a measurement rather than
a guess.

    python scripts/data/tune_detection.py --root data/surrey --limit 400
"""

import argparse
import collections
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEPTH_HINT = re.compile(r"(^|[_\-/])depth", re.I)


def pad(image, fraction: float):
    """Add a border of `fraction` of the longer side, replicating edges."""
    if fraction <= 0:
        return image
    border = int(max(image.shape[:2]) * fraction)
    return cv2.copyMakeBorder(
        image, border, border, border, border, cv2.BORDER_REPLICATE
    )


def upscale(image, factor: float):
    if factor == 1:
        return image
    return cv2.resize(
        image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC
    )


# Each variant: a name, the padding fraction, the upscale factor, and the
# detection confidence. Ordered cheapest first.
VARIANTS = [
    ("baseline",            0.0, 1, 0.5),
    ("pad 25%",             0.25, 1, 0.5),
    ("pad 50%",             0.5, 1, 0.5),
    ("upscale 3x",          0.0, 3, 0.5),
    ("pad 25% + 3x",        0.25, 3, 0.5),
    ("pad 50% + 3x",        0.5, 3, 0.5),
    ("pad 50% + 4x",        0.5, 4, 0.5),
    ("pad 50% + 3x, conf .3", 0.5, 3, 0.3),
    ("pad 50% + 3x, conf .2", 0.5, 3, 0.2),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=400,
                    help="Images to test per variant (strided across the set)")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    images = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES and not DEPTH_HINT.search(p.name)
    )
    if not images:
        print(f"No colour images under {root}")
        return 1

    if args.limit < len(images):
        stride = len(images) // args.limit
        images = images[::stride][: args.limit]

    # Load once; every variant sees exactly the same photographs.
    loaded = []
    for path in images:
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame is None or frame.dtype != np.uint8 or frame.ndim != 3:
            continue
        letter = path.parent.name
        loaded.append((letter, frame))

    print(f"{len(loaded)} images, {len(set(l for l, _ in loaded))} classes\n")
    print(f"{'variant':<26}{'detected':>10}{'rate':>9}{'worst class':>22}")
    print("-" * 67)

    import mediapipe as mp

    results = []

    for name, pad_frac, scale, conf in VARIANTS:
        hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=conf,
        )

        seen = collections.Counter()
        hit = collections.Counter()

        for letter, frame in loaded:
            seen[letter] += 1
            prepared = upscale(pad(frame, pad_frac), scale)
            rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
            if hands.process(rgb).multi_hand_landmarks:
                hit[letter] += 1

        hands.close()

        total = sum(seen.values())
        found = sum(hit.values())
        rate = found / max(total, 1)

        worst_letter = min(seen, key=lambda c: hit[c] / max(seen[c], 1))
        worst_rate = hit[worst_letter] / max(seen[worst_letter], 1)

        results.append((name, rate, worst_letter, worst_rate))
        print(
            f"{name:<26}{found:>6}/{total:<4}{rate:>8.1%}"
            f"{worst_letter + ' ' + format(worst_rate, '.0%'):>22}"
        )

    print("-" * 67)

    best = max(results, key=lambda r: r[1])
    print(f"\nBest overall: {best[0]} at {best[1]:.1%}")

    # A high mean with one collapsed class is worse than a slightly lower
    # mean that holds everywhere, because the gap becomes class imbalance.
    fairest = max(results, key=lambda r: r[3])
    if fairest[0] != best[0]:
        print(f"Most even across classes: {fairest[0]}, "
              f"worst class {fairest[2]} at {fairest[3]:.0%}")
        print("\nPrefer the even one if the rates are close. A variant that")
        print("misses one handshape systematically skews the training set.")

    return 0


if __name__ == "__main__":
    sys.exit(main())