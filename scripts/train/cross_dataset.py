"""
scripts/train/cross_dataset_test.py — Does the Surrey model survive your data?

The leave-one-signer-out result (91.82%) holds out people, but every image
still comes from one dataset: one camera, one room, one capture protocol,
one extractor. This measures the harder thing — a model trained entirely on
Surrey, evaluated on recordings it has never seen from a different person,
camera, room and MediaPipe pipeline.

That number will be lower. It is also the one that answers "does this work
outside the lab", which most comparable projects cannot answer at all.

Three figures, each a different question:

  recorded -> recorded    template recall          (the old 98.96%)
  Surrey   -> Surrey      unseen signer            (91.82% +- 2.83%)
  Surrey   -> recorded    unseen signer AND setup  (this script)

The recorded set is scanned for whatever layout it uses: .npy frames or
sequences nested under class folders, or a single .npz. Only the 24 static
letters Surrey covers are compared — j and z involve motion, and the word
gestures are not in the static model's label set.

    python scripts/train/cross_dataset_test.py \
        --model models/static/static_letters.h5 \
        --labels models/static/labels.json \
        --recorded data/DATASET_NORM
"""

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

FEATURE_DIM = 126


def load_recorded(root: pathlib.Path):
    """
    Find feature vectors however this dataset happens to be stored.

    Handles a single .npz, and .npy files nested under class folders holding
    either one frame (126,) or a sequence (frames, 126). The class is the
    folder directly beneath the root, which is the convention these
    recordings use.
    """
    if root.is_file() and root.suffix == ".npz":
        data = np.load(root, allow_pickle=True)
        return data["X"], np.array([str(v) for v in data["y"]])

    vectors, labels = [], []

    for path in sorted(root.rglob("*.npy")):
        relative = path.relative_to(root).parts
        if len(relative) < 2:
            continue                      # no class folder above the file
        label = relative[0]

        try:
            arr = np.load(path)
        except Exception:
            continue

        arr = np.asarray(arr, dtype=np.float32)

        if arr.ndim == 1 and arr.shape[0] == FEATURE_DIM:
            frames = [arr]
        elif arr.ndim == 2 and arr.shape[1] == FEATURE_DIM:
            frames = list(arr)
        else:
            continue                      # not a 126-dim feature file

        for frame in frames:
            vectors.append(frame)
            labels.append(label)

    if not vectors:
        return np.empty((0, FEATURE_DIM), dtype=np.float32), np.array([])

    return np.array(vectors, dtype=np.float32), np.array(labels)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/static/static_letters.h5")
    ap.add_argument("--labels", default="models/static/labels.json")
    ap.add_argument("--recorded", default="data/DATASET_NORM")
    ap.add_argument("--out", default="Documents/evaluation")
    ap.add_argument("--swap", action="store_true",
                    help="Exchange the left and right hand blocks of each "
                         "recorded vector before predicting. Use if the "
                         "recordings were built with the opposite slot "
                         "convention to the model's training data.")
    args = ap.parse_args()

    labels = json.loads(pathlib.Path(args.labels).read_text())
    index = {label: i for i, label in enumerate(labels)}

    root = pathlib.Path(args.recorded)
    if not root.exists():
        print(f"No such path: {root}")
        return 1

    X, y_raw = load_recorded(root)
    if len(X) == 0:
        print(f"Found no 126-dim feature files under {root}.")
        print("Expected .npy files nested under class folders, or an .npz")
        print("holding X and y. Check the path.")
        return 1

    found = sorted(set(y_raw.tolist()))
    print(f"Loaded {len(X)} frames from {root}")
    print(f"Classes present: {' '.join(found)}\n")

    # Surrey is lower case; the recordings may not be. Match on case-folded
    # single characters, and report what could not be compared.
    mapped, keep = [], []
    unmatched = collections.Counter()

    for i, raw in enumerate(y_raw):
        key = str(raw).strip().lower()
        if key in index:
            mapped.append(index[key])
            keep.append(i)
        else:
            unmatched[str(raw)] += 1

    if unmatched:
        print("Not in the static model's label set, so excluded:")
        for label, count in unmatched.most_common():
            print(f"  {label:<14}{count:>7} frames")
        print("  (j and z involve motion; word gestures are not static letters)\n")

    if not keep:
        print("Nothing overlaps with the 24 static letters. Cannot compare.")
        return 1

    X = X[keep]
    y = np.array(mapped, dtype=np.int64)

    if args.swap:
        # Exchange the two 63-value hand blocks. The parity measurement
        # showed Hands and Holistic label handedness in mirror image of
        # each other; a model trained on one convention reads the other's
        # vectors as an empty hand beside a populated one.
        X = np.concatenate([X[:, 63:], X[:, :63]], axis=1)
        print("Hand slots swapped.\n")

    print(f"Comparing on {len(X)} frames across "
          f"{len(set(y.tolist()))} letters.\n")

    from tensorflow import keras

    model = keras.models.load_model(args.model)
    probs = model.predict(X, verbose=0)
    predicted = probs.argmax(axis=1)

    accuracy = float((predicted == y).mean())
    confidence = float(probs.max(axis=1).mean())

    print(f"Frame-level accuracy  {accuracy:.2%}")
    print(f"Mean confidence       {confidence:.2%}")

    # Per-letter, worst first — this is where the interesting part is.
    print("\nPer-letter:")
    per_class = {}
    for cls in sorted(set(y.tolist())):
        mask = y == cls
        rate = float((predicted[mask] == cls).mean())
        per_class[labels[cls]] = rate

    for label, rate in sorted(per_class.items(), key=lambda kv: kv[1]):
        bar = "#" * int(rate * 24)
        print(f"  {label:<4}{rate:>7.1%}  {bar}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cross_dataset.json").write_text(json.dumps({
        "train": "Surrey (5 signers)",
        "test": str(root),
        "frames": int(len(X)),
        "frame_accuracy": accuracy,
        "mean_confidence": confidence,
        "per_letter": per_class,
        "excluded": {k: int(v) for k, v in unmatched.items()},
    }, indent=2))

    print(f"\nWrote {out}/cross_dataset.json")

    print("\nFor the results chapter:")
    print(f"  recorded -> recorded   98.96%           template recall")
    print(f"  Surrey   -> Surrey     91.82% +- 2.83%  unseen signer")
    print(f"  Surrey   -> recorded   {accuracy:.2%}           unseen signer and setup")

    print()
    if accuracy >= 0.7:
        print("Holds up across datasets. That is a strong generalisation")
        print("result and unusual to be able to report at all.")
    elif accuracy >= 0.4:
        print("Degrades across datasets. Expected, and worth reporting: the")
        print("gap between this and 91.82% is the cost of changing camera,")
        print("room and capture protocol.")
    else:
        print("Does not transfer. Worth checking before concluding: the")
        print("recordings were extracted with Holistic, the Surrey features")
        print("with Hands. Parity was measured at 0.9995 once hand slots")
        print("were swapped — if these recordings predate that fix, their")
        print("slots may be reversed relative to the model's expectation.")
        print("Re-run with --swap to test that.")

    return 0


if __name__ == "__main__":
    sys.exit(main())