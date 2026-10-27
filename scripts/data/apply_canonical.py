"""
scripts/data/apply_canonical.py — Canonicalise an extracted feature file

Moves every one-handed sign into the same block, so a model learns each
handshape once rather than once per hand. See the canonicalise() docstring
in signbridge/normalization.py for why.

Reports the before and after distribution, and backs up the original.

    python scripts/data/apply_canonical.py data/processed/surrey.npz

For the recorded dataset, which is a tree of .npy files rather than a single
archive, --recorded writes a canonicalised .npz instead of editing in place:

    python scripts/data/apply_canonical.py data/DATASET_NORM \
        --recorded --out data/processed/recorded_canonical.npz
"""

import argparse
import collections
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from signbridge.normalization import canonicalise_batch  # noqa: E402

FEATURE_DIM = 126


def slots(X: np.ndarray) -> collections.Counter:
    left = np.abs(X[:, :63]).sum(axis=1) > 1e-6
    right = np.abs(X[:, 63:]).sum(axis=1) > 1e-6
    return collections.Counter({
        "left only": int((left & ~right).sum()),
        "right only": int((right & ~left).sum()),
        "both": int((left & right).sum()),
        "neither": int((~left & ~right).sum()),
    })


def show(title: str, counts: collections.Counter, total: int) -> None:
    print(f"  {title}")
    for key in ("left only", "right only", "both", "neither"):
        share = counts[key] / max(total, 1)
        print(f"    {key:<12}{counts[key]:>8}{share:>8.1%}")


def load_tree(root: pathlib.Path):
    """Feature vectors and labels from a tree of .npy files."""
    vectors, labels = [], []

    for npy in sorted(root.rglob("*.npy")):
        parts = npy.relative_to(root).parts
        if len(parts) < 2:
            continue
        try:
            arr = np.asarray(np.load(npy), dtype=np.float32)
        except Exception:
            continue

        if arr.ndim == 1 and arr.shape[0] == FEATURE_DIM:
            frames = [arr]
        elif arr.ndim == 2 and arr.shape[1] == FEATURE_DIM:
            frames = list(arr)
        else:
            continue

        for frame in frames:
            vectors.append(frame)
            labels.append(parts[0])

    return (
        np.array(vectors, dtype=np.float32) if vectors else np.empty((0, FEATURE_DIM)),
        np.array(labels),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help=".npz file, or a folder with --recorded")
    ap.add_argument("--recorded", action="store_true",
                    help="Treat the path as a tree of .npy files")
    ap.add_argument("--out", help="Where to write (required with --recorded)")
    args = ap.parse_args()

    path = pathlib.Path(args.path)
    if not path.exists():
        print(f"No such path: {path}")
        return 1

    if args.recorded:
        if not args.out:
            print("--out is required with --recorded")
            return 1

        X, y = load_tree(path)
        if len(X) == 0:
            print(f"No 126-dim feature files under {path}")
            return 1

        print(f"{path}  —  {len(X)} vectors\n")
        show("before", slots(X), len(X))

        X = canonicalise_batch(X)
        show("after", slots(X), len(X))

        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, X=X, y=y)
        print(f"\nWrote {out}")
        return 0

    data = np.load(path, allow_pickle=True)
    X = data["X"]

    print(f"{path}  —  {len(X)} vectors\n")
    show("before", slots(X), len(X))

    canonical = canonicalise_batch(X)
    show("after", slots(canonical), len(X))

    backup = path.with_suffix(".npz.before-canonical")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"\nBacked up to {backup}")

    np.savez_compressed(
        path,
        X=canonical,
        **{k: data[k] for k in data.files if k != "X"},
    )
    print(f"Wrote {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())