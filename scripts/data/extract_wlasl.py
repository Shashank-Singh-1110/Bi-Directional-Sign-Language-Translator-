import argparse
import collections
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from signbridge.normalization import (  # noqa: E402
    build_features, canonicalise, empty_hand,
)

SEQUENCE_LENGTH = 30
HAND_DIM = 126
LOCATION_DIM = 6

# MediaPipe Pose landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

EPSILON = 1e-6


def to_63(landmarks) -> np.ndarray:
    if landmarks is None:
        return empty_hand()
    return np.array(
        [[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32
    ).flatten()


def wrist_of(landmarks) -> np.ndarray | None:
    """Landmark 0 — the wrist — in image coordinates."""
    if landmarks is None:
        return None
    p = landmarks.landmark[0]
    return np.array([p.x, p.y, p.z], dtype=np.float32)


def body_frame(pose):
    """
    Origin and scale for expressing hand position relative to the body.

    Origin is the midpoint of the shoulders; scale is the shoulder width.
    Returns None when the pose is missing or degenerate, in which case the
    location block is zeroed rather than filled with nonsense.
    """
    if pose is None:
        return None

    left = pose.landmark[LEFT_SHOULDER]
    right = pose.landmark[RIGHT_SHOULDER]

    # Visibility is MediaPipe's own confidence that the joint was seen.
    if min(getattr(left, "visibility", 1.0),
           getattr(right, "visibility", 1.0)) < 0.5:
        return None

    a = np.array([left.x, left.y, left.z], dtype=np.float32)
    b = np.array([right.x, right.y, right.z], dtype=np.float32)

    width = float(np.linalg.norm(a[:2] - b[:2]))
    if width < EPSILON:
        return None

    return (a + b) / 2.0, width


def location_block(results) -> np.ndarray:
    """
    Six values: each wrist in the body frame, or zeros where unavailable.

    Zeros for a missing hand are consistent with how the hand-shape blocks
    already represent absence.
    """
    out = np.zeros(LOCATION_DIM, dtype=np.float32)

    frame = body_frame(results.pose_landmarks)
    if frame is None:
        return out

    origin, scale = frame

    for i, marks in enumerate(
        (results.left_hand_landmarks, results.right_hand_landmarks)
    ):
        wrist = wrist_of(marks)
        if wrist is None:
            continue
        out[i * 3:(i + 1) * 3] = (wrist - origin) / scale

    return out


def canonicalise_with_location(hands_126, location_6):
    """
    Canonicalise hand shape, moving the matching location values with it.

    The hand-shape blocks and the location blocks have to stay aligned: if
    a lone left hand is moved into the right slot, its position must move
    too, or the model sees one hand's shape beside the other's location.
    """
    left_present = np.abs(hands_126[:63]).sum() > EPSILON
    right_present = np.abs(hands_126[63:]).sum() > EPSILON

    hands = canonicalise(hands_126)
    location = location_6.copy()

    if left_present and not right_present:
        location[3:6] = location_6[0:3]
        location[0:3] = 0.0

    return hands, location


def sample_indices(total: int, want: int) -> list[int]:
    """`want` frame positions spread evenly over `total` frames."""
    if total <= 0:
        return []
    if total <= want:
        return list(range(total)) + [total - 1] * (want - total)
    return [int(round(i * (total - 1) / (want - 1))) for i in range(want)]


def read_sequence(model, path, start, end, use_location):
    """One clip to a (30, D) array, or None with a reason."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, "unreadable"

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return None, "empty"

    first = max(0, start - 1) if start and start > 0 else 0
    last = len(frames) if (end is None or end <= 0) else min(end, len(frames))
    clip = frames[first:last]

    if len(clip) < 4:
        return None, "too short"

    sequence, with_hands = [], 0

    for i in sample_indices(len(clip), SEQUENCE_LENGTH):
        rgb = cv2.cvtColor(clip[i], cv2.COLOR_BGR2RGB)
        res = model.process(rgb)

        hands = build_features(
            to_63(res.left_hand_landmarks), to_63(res.right_hand_landmarks)
        )

        if res.left_hand_landmarks or res.right_hand_landmarks:
            with_hands += 1

        if use_location:
            hands, location = canonicalise_with_location(
                hands, location_block(res)
            )
            sequence.append(np.concatenate([hands, location]))
        else:
            sequence.append(canonicalise(hands))

    # A clip where hands are rarely found carries no usable signal, and
    # keeping it would teach the model that this word looks like nothing.
    if with_hands < SEQUENCE_LENGTH * 0.4:
        return None, "few hands"

    return np.array(sequence, dtype=np.float32), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--features", choices=("hands", "location"),
                    default="location",
                    help="hands = 126 dims; location = 132 (default)")
    args = ap.parse_args()

    use_location = args.features == "location"

    entries = json.loads(pathlib.Path(args.json).read_text())
    videos = pathlib.Path(args.videos)

    by_count = sorted(
        entries, key=lambda e: len(e.get("instances", [])), reverse=True
    )
    chosen = by_count[: args.top]
    glosses = sorted(e["gloss"] for e in chosen)

    # Index every video by file stem, searching recursively. The start_kit
    # writes a flat videos/ folder; the Hugging Face mirror spreads files
    # across data_0/, data_1/ and so on. Both are keyed by WLASL video id,
    # so one index serves either — and the mirror carries no signer_id of
    # its own, which is why the original JSON remains the source of truth
    # for gloss, signer and frame range.
    index = {}
    for path in videos.rglob("*"):
        if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".avi", ".mov"}:
            index.setdefault(path.stem, path)

    if not index:
        print(f"No video files found under {videos}.")
        return 1

    tasks = []
    for entry in chosen:
        for inst in entry.get("instances", []):
            path = index.get(str(inst["video_id"]))
            if path is not None:
                tasks.append((entry["gloss"], inst, path))

    if not tasks:
        print(f"Found {len(index)} video files under {videos}, but none of")
        print("their names match a video_id in the JSON. Check that the")
        print("files are named by WLASL video id.")
        return 1

    present = len(tasks)
    total_listed = sum(len(e.get("instances", [])) for e in chosen)
    print(f"{len(glosses)} glosses · {present} of {total_listed} videos present "
          f"({present / max(total_listed, 1):.0%})")
    print(f"Features: {args.features} "
          f"({HAND_DIM + (LOCATION_DIM if use_location else 0)} dims)")

    if args.limit and args.limit < len(tasks):
        stride = len(tasks) // args.limit
        tasks = tasks[::stride][: args.limit]
        print(f"Sampling {len(tasks)} clips spread across the set.")

    print()

    import mediapipe as mp

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    sequences, labels, signers = [], [], []
    skipped = collections.Counter()
    no_body = 0
    started = time.time()

    for i, (gloss, inst, path) in enumerate(tasks, 1):
        seq, why = read_sequence(
            holistic, path,
            inst.get("frame_start", 0), inst.get("frame_end", -1),
            use_location,
        )

        if seq is None:
            skipped[why] += 1
        else:
            sequences.append(seq)
            labels.append(gloss)
            signers.append(str(inst.get("signer_id", "unknown")))

            # How often the body frame was unavailable, which would leave
            # the location block zeroed and the feature useless.
            if use_location and np.abs(seq[:, HAND_DIM:]).sum() < EPSILON:
                no_body += 1

        if i % 25 == 0 or i == len(tasks):
            rate = i / max(time.time() - started, 1e-6)
            left = (len(tasks) - i) / max(rate, 1e-6)
            print(f"  {i}/{len(tasks)}  kept {len(sequences)}  "
                  f"{rate:.1f}/s  ~{left / 60:.0f} min left")

    holistic.close()

    if not sequences:
        print("\nNo usable sequences.")
        return 1

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels)
    s = np.array(signers)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, signer=s, features=args.features)

    print(f"\nWrote {out}")
    print(f"  {X.shape[0]} sequences of shape {X.shape[1:]}")
    print(f"  {len(set(labels))} glosses · {len(set(signers))} signers")

    if skipped:
        print("\nSkipped:")
        for why, count in skipped.most_common():
            print(f"  {why:<14}{count:>6}")

    if use_location:
        print(f"\nBody frame unavailable in {no_body} of {len(sequences)} "
              f"sequences ({no_body / len(sequences):.0%})")
        if no_body > len(sequences) * 0.2:
            print("Above 20%. The location block is zeroed for those, so the")
            print("feature is doing less work than its dimensionality implies.")

    counts = collections.Counter(labels)
    print(f"\nMedian per gloss: {int(np.median(list(counts.values())))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())