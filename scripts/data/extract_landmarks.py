"""
scripts/data/extract_landmarks.py — Images to normalised landmark features

Walks a dataset of images, runs MediaPipe Hands over each one, and writes the
126-dim normalised feature vectors the classifier consumes.

Three things this does that a naive extractor would not:

1. Swaps the hand slots. verify_extractor_parity.py measured Hands against
   Holistic at 0.0000 similarity as-is and 0.9995 swapped — the geometry
   agrees, the handedness labels are mirrored. Without the swap every vector
   lands in the wrong half of the feature space.

2. Records a signer id per image, not just a class label. A split that puts
   the same person in train and test measures template recall, which is the
   failure already diagnosed in the project's own recorded dataset. Signer
   ids are what make a signer-disjoint split possible.

3. Skips depth maps. The Surrey set ships a depth frame beside every colour
   frame. They are images by file extension but hold distance values, not
   pixels, so MediaPipe cannot read them and they would otherwise halve the
   reported detection rate.

The directory layout is inferred by default: the level whose values look
like class names becomes the class, and a shallower level with few distinct
values becomes the signer. Both can be set explicitly if the guess is wrong.

    python scripts/data/extract_landmarks.py --root data/surrey --dry-run

    python scripts/data/extract_landmarks.py \
        --root data/surrey --out data/processed/surrey.npz
"""

import argparse
import collections
import pathlib
import re
import sys
import time

# Python puts this file's directory on sys.path, not the directory you ran
# from, so the repo root has to be added explicitly for `signbridge` to
# import. Three levels up: scripts/data/this_file.py -> repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from signbridge.normalization import build_features, empty_hand  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Depth frames sit beside the colour ones and are named for what they hold.
DEPTH_HINT = re.compile(r"(^|[_\-/])depth", re.I)
COLOUR_HINT = re.compile(r"(^|[_\-/])(color|colour|rgb)", re.I)

# Defaults measured by tune_detection.py on Surrey, where the images are
# ~110x120 crops with the hand touching every edge:
#
#   baseline                 67.5%   worst class 25%
#   pad 50%                  93.0%   worst class 68%
#   upscale 3x alone         52.0%   worst class  5%   <- worse than nothing
#   pad 50% + 3x, conf 0.2   94.5%   worst class 80%   <- chosen
#
# Padding is what matters: the detector needs somewhere to place a bounding
# box. Upscaling only helps once that margin exists. Both are affine and
# uniform, so wrist-origin normalisation divides them back out — the
# landmarks move, the feature vector does not.
PAD_FRACTION = 0.5
UPSCALE = 3
DETECTION_CONFIDENCE = 0.2


def find_images(root: pathlib.Path, keep_depth: bool) -> list[pathlib.Path]:
    """
    Every readable colour image under root.

    Depth frames are excluded by name. A second, content-based check happens
    at read time, because a dataset that names its files differently would
    otherwise slip through.
    """
    out = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not keep_depth and DEPTH_HINT.search(path.name):
            continue
        out.append(path)
    return out


def is_depth_frame(image) -> bool:
    """
    A depth map rather than a photograph.

    Kinect depth frames are single-channel 16-bit. A colour photograph read
    by OpenCV is three-channel 8-bit, so the test is unambiguous.
    """
    return image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3


def levels(root: pathlib.Path, images: list[pathlib.Path], cap: int = 8000):
    """
    Distinct values at each directory level, indexed from the file up.

    Sampled with a stride across the whole list rather than from the head.
    Paths arrive sorted, so the first N of a large set all sit inside the
    first top-level folder — which made a five-signer dataset look like it
    had one.
    """
    found = collections.defaultdict(set)
    stride = max(1, len(images) // cap)

    for path in images[::stride]:
        parts = path.relative_to(root).parts[:-1]
        for i in range(1, min(len(parts), 5) + 1):
            found[-i].add(parts[-i])
    return found


def looks_like_classes(values: set[str]) -> bool:
    """
    Class folders in a fingerspelling set are single letters or short words.

    Used only to pick a default; an explicit --class-depth always wins.
    """
    if not 2 <= len(values) <= 500:
        return False
    short = sum(1 for v in values if len(v) <= 3)
    return short / len(values) > 0.8


def infer_depths(found) -> tuple[int | None, int | None]:
    """
    Guess which level holds the class and which holds the signer.

    The class level is the deepest one whose values look like class names.
    The signer level is the shallowest other level with a small number of
    distinct values — subjects, sessions, anything a split must not cross.
    """
    class_depth = next(
        (d for d in sorted(found, reverse=True) if looks_like_classes(found[d])),
        None,
    )

    signer_depth = None
    if class_depth is not None:
        candidates = [
            d for d in found
            if d != class_depth and 2 <= len(found[d]) <= 50
        ]
        if candidates:
            signer_depth = max(candidates)   # the level just above the class

    return class_depth, signer_depth


def component(path: pathlib.Path, root: pathlib.Path, depth: int) -> str:
    """A path component relative to the root, indexed from the file upward."""
    parts = path.relative_to(root).parts[:-1]
    try:
        return parts[depth]
    except IndexError:
        return ""


def describe(root, images, found, class_depth, signer_depth) -> None:
    print(f"{len(images)} colour images under {root}\n")

    for depth in sorted(found, reverse=True):
        values = sorted(found[depth])
        tag = ""
        if depth == class_depth:
            tag = "  <- class"
        elif depth == signer_depth:
            tag = "  <- signer"
        preview = ", ".join(values[:12])
        more = f" … +{len(values) - 12}" if len(values) > 12 else ""
        print(f"  depth {depth:>2}: {len(values):>4} distinct — {preview}{more}{tag}")

    print("\nExample paths:")
    for path in images[:5]:
        print(f"  {path.relative_to(root)}")

    print()
    if class_depth is None:
        print("Could not identify a class level. Pass --class-depth.")
    else:
        print(f"Inferred --class-depth {class_depth}", end="")
        print(
            f" --signer-depth {signer_depth}" if signer_depth is not None
            else "  (no signer level found — pass --signer-depth)"
        )


def prepare(image, pad_fraction: float, scale: float):
    """
    Give the detector margin and resolution to work with.

    Tight crops defeat MediaPipe's palm detector: with the hand against
    every edge there is nowhere to put a bounding box. Replicating the
    border fixes that, and upscaling afterwards recovers detail lost to the
    Kinect's small frames.
    """
    if pad_fraction > 0:
        border = int(max(image.shape[:2]) * pad_fraction)
        image = cv2.copyMakeBorder(
            image, border, border, border, border, cv2.BORDER_REPLICATE
        )
    if scale != 1:
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    return image


def to_63(landmarks) -> np.ndarray:
    return np.array(
        [[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32
    ).flatten()


def extract(model, rgb) -> np.ndarray | None:
    """
    One image to a 126-dim feature vector, or None if no hand was found.

    Hands labels handedness for the image as presented. The measured parity
    check showed that label is the mirror of Holistic's, so "Left" fills the
    right slot and vice versa.
    """
    res = model.process(rgb)
    if not res.multi_hand_landmarks:
        return None

    left, right = empty_hand(), empty_hand()

    for marks, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
        flat = to_63(marks)
        if handed.classification[0].label == "Left":
            right = flat          # mirrored: see the parity measurement
        else:
            left = flat

    return build_features(left, right)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="Dataset root folder")
    ap.add_argument("--out", help="Where to write the .npz")
    ap.add_argument("--class-depth", type=int,
                    help="Path level holding the class label (inferred if omitted)")
    ap.add_argument("--signer-depth", type=int,
                    help="Path level holding the signer id (inferred if omitted)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show the structure found and stop")
    ap.add_argument("--limit", type=int,
                    help="Only process this many images, for a quick check")
    ap.add_argument("--keep-depth", action="store_true",
                    help="Do not skip files named like depth maps")
    ap.add_argument("--pad", type=float, default=PAD_FRACTION,
                    help=f"Border as a fraction of the longer side "
                         f"(default {PAD_FRACTION})")
    ap.add_argument("--upscale", type=float, default=UPSCALE,
                    help=f"Resize factor after padding (default {UPSCALE})")
    ap.add_argument("--confidence", type=float, default=DETECTION_CONFIDENCE,
                    help=f"Minimum detection confidence "
                         f"(default {DETECTION_CONFIDENCE})")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    if not root.is_dir():
        print(f"Not a folder: {root}")
        return 1

    images = find_images(root, args.keep_depth)
    if not images:
        print(f"No colour images found under {root}")
        return 1

    found = levels(root, images)
    guess_class, guess_signer = infer_depths(found)

    class_depth = args.class_depth if args.class_depth is not None else guess_class
    signer_depth = (
        args.signer_depth if args.signer_depth is not None else guess_signer
    )

    if args.dry_run:
        describe(root, images, found, class_depth, signer_depth)
        return 0

    if class_depth is None:
        print("Could not identify the class level. Run --dry-run and pass")
        print("--class-depth explicitly.")
        return 1

    if not args.out:
        print("--out is required unless --dry-run")
        return 1

    if signer_depth is None:
        print("WARNING: no signer level. Every image will be attributed to")
        print("one signer, and a signer-disjoint split becomes impossible —")
        print("which is the point of this extraction.\n")

    if args.limit and args.limit < len(images):
        # Strided, not the head. Paths are sorted, so the first N all come
        # from one signer's first letter and would report a detection rate
        # for that one case rather than the dataset.
        stride = len(images) // args.limit
        images = images[::stride][: args.limit]
        print(f"Sampling {len(images)} images spread across the set.\n")

    import mediapipe as mp

    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=args.confidence,
    )

    print(
        f"Preprocessing: pad {args.pad:.0%}, upscale {args.upscale}x, "
        f"confidence {args.confidence}\n"
    )

    features, labels, signers, kept_paths = [], [], [], []
    seen, missed = collections.Counter(), collections.Counter()
    skipped_depth = 0
    unreadable = 0
    started = time.time()

    for i, path in enumerate(images, 1):
        label = component(path, root, class_depth)
        signer = (
            component(path, root, signer_depth)
            if signer_depth is not None else "unknown"
        )

        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            unreadable += 1
            continue

        # Content check, for datasets that do not name depth files clearly.
        if is_depth_frame(frame):
            skipped_depth += 1
            continue

        seen[label] += 1

        prepared = prepare(frame, args.pad, args.upscale)
        vec = extract(hands, cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB))
        if vec is None:
            missed[label] += 1
            continue

        features.append(vec)
        labels.append(label)
        signers.append(signer)
        kept_paths.append(str(path.relative_to(root)))

        if i % 250 == 0 or i == len(images):
            rate = i / max(time.time() - started, 1e-6)
            print(f"  {i}/{len(images)}  kept {len(features)}  {rate:.0f}/s")

    hands.close()

    if skipped_depth:
        print(f"\nSkipped {skipped_depth} depth frames (not photographs).")
    if unreadable:
        print(f"Skipped {unreadable} unreadable files.")

    if not features:
        print("\nNo hand was detected in any image.")
        return 1

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)
    s = np.array(signers)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, signer=s, paths=np.array(kept_paths))

    print(f"\nWrote {out}")
    print(f"  {X.shape[0]} vectors of {X.shape[1]} values")
    print(f"  {len(set(labels))} classes, {len(set(signers))} signers")

    # Detection rate per class. A class the extractor mostly fails on will
    # be under-represented in training, and that has to be visible now
    # rather than discovered as a confusing accuracy result later.
    print("\nLowest detection rates by class:")
    worst = sorted(seen, key=lambda c: (seen[c] - missed[c]) / max(seen[c], 1))
    for label in worst[:10]:
        total = seen[label]
        kept = total - missed[label]
        print(f"  {label:<12} {kept:>6}/{total:<6} {kept / max(total, 1):6.1%}")
    if len(worst) > 10:
        print(f"  … {len(worst) - 10} further classes")

    graded = sum(seen.values())
    overall = len(features) / max(graded, 1)
    print(f"\nOverall detection rate {overall:.1%} of {graded} photographs")
    if overall < 0.8:
        print("Below 80%. Inspect a few failures before training — a")
        print("systematic miss will skew the class balance.")

    print("\nSigner distribution:")
    for signer, count in sorted(collections.Counter(signers).items()):
        print(f"  {signer:<12} {count:>7}")

    return 0


if __name__ == "__main__":
    sys.exit(main())