"""
scripts/data/check_slots.py — Which hand slot does each dataset populate?

The slot swap applied at extraction was calibrated on a mirrored webcam
feed. Surrey's images come from a Kinect and are not mirrored, so the same
correction may not apply to them — and if it does not, the model was trained
on vectors with the hand in the opposite block from where the deployed
system puts it.

Reasoning about mirroring in the abstract is how the Android client lost
three weeks. This just looks.

A 126-dim vector is [left_hand(63), right_hand(63)]. For a one-handed sign
exactly one block should be non-zero. This reports which, per dataset, so
the two can be compared directly.

    python scripts/data/check_slots.py \
        data/processed/surrey.npz data/DATASET_NORM
"""

import argparse
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

FEATURE_DIM = 126


def occupancy(X: np.ndarray) -> collections.Counter:
    """Count which blocks carry data across a set of vectors."""
    left = np.abs(X[:, :63]).sum(axis=1) > 1e-6
    right = np.abs(X[:, 63:]).sum(axis=1) > 1e-6

    counts = collections.Counter()
    counts["left only"] = int((left & ~right).sum())
    counts["right only"] = int((right & ~left).sum())
    counts["both"] = int((left & right).sum())
    counts["neither"] = int((~left & ~right).sum())
    return counts


def load(path: pathlib.Path) -> np.ndarray:
    """Feature vectors from either an .npz or a tree of .npy files."""
    if path.is_file() and path.suffix == ".npz":
        return np.load(path, allow_pickle=True)["X"]

    vectors = []
    for npy in sorted(path.rglob("*.npy")):
        try:
            arr = np.asarray(np.load(npy), dtype=np.float32)
        except Exception:
            continue
        if arr.ndim == 1 and arr.shape[0] == FEATURE_DIM:
            vectors.append(arr)
        elif arr.ndim == 2 and arr.shape[1] == FEATURE_DIM:
            vectors.extend(arr)

    return np.array(vectors, dtype=np.float32) if vectors else np.empty((0, FEATURE_DIM))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="One or more .npz files or folders")
    args = ap.parse_args()

    summary = {}

    for raw in args.paths:
        path = pathlib.Path(raw)
        if not path.exists():
            print(f"No such path: {path}\n")
            continue

        X = load(path)
        if len(X) == 0:
            print(f"{path}: no 126-dim vectors found\n")
            continue

        counts = occupancy(X)
        total = sum(counts.values())

        print(f"{path}  —  {total} vectors")
        for key in ("left only", "right only", "both", "neither"):
            share = counts[key] / total
            bar = "#" * int(share * 30)
            print(f"  {key:<12}{counts[key]:>9}{share:>8.1%}  {bar}")
        print()

        dominant = max(
            ("left only", "right only"), key=lambda k: counts[k]
        )
        summary[str(path)] = dominant

    if len(summary) >= 2:
        values = list(summary.values())
        print("-" * 52)
        for name, slot in summary.items():
            print(f"  {name:<36}{slot}")
        print("-" * 52)

        if len(set(values)) == 1:
            print("\nBoth datasets populate the SAME slot.")
            print("No swap is needed between them. If transfer is poor, the")
            print("cause is elsewhere — a genuine domain gap, not slots.")
        else:
            print("\nThe datasets populate OPPOSITE slots.")
            print("A model trained on one reads the other's vectors as an")
            print("empty hand beside a populated one. Fix this at extraction")
            print("so training and inference share one convention, rather")
            print("than swapping at evaluation time.")

    return 0


if __name__ == "__main__":
    sys.exit(main())