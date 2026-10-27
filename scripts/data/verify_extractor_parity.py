"""
scripts/data/verify_extractor_parity.py — Are Hands and Holistic interchangeable?

Public ASL alphabet datasets are cropped hand images with no body in frame.
MediaPipe Holistic locates hands via the pose model, so it fails on those
images. The alternative is mediapipe.solutions.hands, which detects hands
directly — but that is a different extractor from the one that produced the
training data.

We already know one pair of MediaPipe APIs (Holistic vs the HandLandmarker
Tasks API) produce geometrically incompatible landmarks. So this is measured
before anything is built on it, not assumed.

Run this on frames that contain a visible upper body, because Holistic needs
one. Webcam frames of yourself signing are ideal.

    python scripts/data/verify_extractor_parity.py frames/

Reads every image in the folder. For each, extracts the 126-dim feature
vector twice — once through Holistic, once through Hands — and reports the
cosine similarity between them.

Verdict thresholds come from the Android diagnostic: the classifier needs
~0.997 similarity to behave identically, and 0.775 was demonstrably not
enough.
"""

import argparse
import pathlib
import sys

# Python puts this file's directory on sys.path, not the directory you ran
# from, so the repo root has to be added explicitly for `signbridge` to
# import. Three levels up: scripts/data/this_file.py -> repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from signbridge.normalization import build_features, empty_hand  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Below this, the extractors are not interchangeable and the dataset plan
# has to change. Taken from the Android diagnostic, where 0.775 produced
# wrong classifications and 0.997 was the measured requirement.
SAFE = 0.997
MARGINAL = 0.99


def landmarks_to_63(landmarks) -> np.ndarray:
    """A MediaPipe landmark list to the flat 63-value form."""
    if landmarks is None:
        return empty_hand()
    return np.array(
        [[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32
    ).flatten()


def via_holistic(model, rgb) -> np.ndarray | None:
    """Features the way the training data was produced. None if no hand."""
    res = model.process(rgb)
    if not (res.left_hand_landmarks or res.right_hand_landmarks):
        return None

    return build_features(
        landmarks_to_63(res.left_hand_landmarks),
        landmarks_to_63(res.right_hand_landmarks),
    )


def via_hands(model, rgb) -> np.ndarray | None:
    """
    Features from the Hands solution.

    Hands reports handedness per detection rather than filling fixed slots,
    so we place each detected hand into the slot its label names. That slot
    assignment has to match Holistic's or the vectors cannot agree, which is
    part of what this script is checking.
    """
    res = model.process(rgb)
    if not res.multi_hand_landmarks:
        return None

    left, right = empty_hand(), empty_hand()

    for marks, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
        flat = np.array(
            [[p.x, p.y, p.z] for p in marks.landmark], dtype=np.float32
        ).flatten()

        if handed.classification[0].label == "Left":
            left = flat
        else:
            right = flat

    return build_features(left, right)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


def swap_slots(vec: np.ndarray) -> np.ndarray:
    """Exchange the left and right hand blocks of a 126-dim feature vector."""
    return np.concatenate([vec[63:], vec[:63]])


def describe(vec: np.ndarray) -> str:
    """Which hand slots carry data. Distinguishes empty from populated."""
    left = np.abs(vec[:63]).sum() > 1e-6
    right = np.abs(vec[63:]).sum() > 1e-6
    if left and right:
        return "both"
    if left:
        return "L"
    if right:
        return "R"
    return "none"


def capture(folder: pathlib.Path, count: int) -> None:
    """
    Grab frames from the webcam so there is something to compare.

    Holistic needs a visible upper body, so sit back far enough that your
    head and shoulders are in frame — the same framing that made the live
    demo work.
    """
    folder.mkdir(parents=True, exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Could not open the webcam")

    print(f"Capturing {count} frames into {folder}.")
    print("Sit back so your head and shoulders are visible.")
    print("SPACE saves a frame, Q stops early.\n")

    saved = 0
    while saved < count:
        ok, frame = cam.read()
        if not ok:
            break

        shown = frame.copy()
        cv2.putText(
            shown, f"{saved}/{count}  SPACE save  Q quit",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.imshow("capture", shown)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            path = folder / f"frame_{saved:02d}.png"
            cv2.imwrite(str(path), frame)
            saved += 1
            print(f"  saved {path.name}")
        elif key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"\n{saved} frames saved.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "folder", nargs="?", default="frames",
        help="Folder of images showing a hand and upper body (default: frames)",
    )
    ap.add_argument(
        "--capture", type=int, metavar="N",
        help="Grab N frames from the webcam into the folder first",
    )
    args = ap.parse_args()

    folder = pathlib.Path(args.folder)

    if args.capture:
        capture(folder, args.capture)

    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        print("Run with --capture 10 to grab frames from the webcam first.")
        return 1

    images = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        print(f"No images found in {folder}")
        return 1

    import mediapipe as mp

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    )
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    )

    print(f"{len(images)} images\n")
    print(
        f"{'image':<26}{'holistic':>10}{'hands':>8}"
        f"{'as-is':>9}{'swapped':>10}"
    )
    print("-" * 63)

    scores = []
    swapped_scores = []
    only_hands = 0
    neither = 0
    slot_mismatch = 0

    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h_vec = via_holistic(holistic, rgb)
        n_vec = via_hands(hands, rgb)

        slots_h = describe(h_vec) if h_vec is not None else "—"
        slots_n = describe(n_vec) if n_vec is not None else "—"

        if h_vec is not None and n_vec is not None:
            # Compared both ways. A near-zero as-is score beside a high
            # swapped score means the geometry agrees and only the slot
            # assignment differs — a relabelling, not an incompatibility.
            direct = cosine(h_vec, n_vec)
            flipped = cosine(h_vec, swap_slots(n_vec))

            scores.append(direct)
            swapped_scores.append(flipped)

            if flipped > direct + 0.3:
                slot_mismatch += 1

            as_is, swapped = f"{direct:.4f}", f"{flipped:.4f}"
        else:
            as_is = swapped = "—"
            if h_vec is None and n_vec is not None:
                only_hands += 1
            elif h_vec is None and n_vec is None:
                neither += 1

        print(
            f"{path.name[:25]:<26}{slots_h:>10}{slots_n:>8}"
            f"{as_is:>9}{swapped:>10}"
        )

    holistic.close()
    hands.close()

    print("-" * 62)

    if not scores:
        print("\nNo image produced a hand from both extractors.")
        print("Holistic needs a visible upper body — try frames where your")
        print("torso and head are in view, not close-ups of a hand.")
        return 1

    arr = np.array(scores)
    flip = np.array(swapped_scores)

    print(f"\nPaired on {len(scores)} of {len(images)} images")
    print(f"  as-is     mean {arr.mean():.4f}   min {arr.min():.4f}")
    print(f"  swapped   mean {flip.mean():.4f}   min {flip.min():.4f}")

    if only_hands:
        print(f"\n{only_hands} images: Hands found a hand, Holistic did not.")
        print("Expected for cropped images — it is the reason for this check.")
    if neither:
        print(f"{neither} images: neither extractor found a hand.")

    # The swapped comparison is what separates a relabelling from a real
    # geometric incompatibility. Judge on whichever orientation is better.
    best, label = (flip, "swapped") if flip.mean() > arr.mean() else (arr, "as-is")

    print()
    if slot_mismatch:
        print(f"SLOT MISMATCH on {slot_mismatch} of {len(scores)} images.")
        print("The two extractors assign the same hand to opposite slots.")
        print("Hands reports handedness for the image as given; a webcam feed")
        print("is mirrored, so its label is the reverse of the signer's own")
        print("hand. Fix by swapping the slots at extraction time.\n")

    if best.min() >= SAFE:
        print(f"VERDICT: interchangeable in the {label} orientation")
        print(f"(min {best.min():.4f} >= {SAFE}).")
        print("Geometry agrees. Use Hands for the public dataset" +
              (", swapping the hand slots." if label == "swapped" else "."))
    elif best.mean() >= MARGINAL:
        print(f"VERDICT: close but not clean ({label}, mean {best.mean():.4f}).")
        print("Usable only if the whole model is retrained on Hands-extracted")
        print("features. Do not mix extractors between training and serving.")
    else:
        print(f"VERDICT: NOT interchangeable ({label}, mean {best.mean():.4f}).")
        print("Same situation as Holistic vs HandLandmarker. Extract training")
        print("data and serve inference with the same extractor, or the")
        print("features the model sees at run time will not match training.")

    return 0


if __name__ == "__main__":
    sys.exit(main())