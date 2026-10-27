"""
scripts/data/fix_slots.py — Flip the hand blocks of an extracted feature file

The slot swap in extract_landmarks.py was calibrated on a mirrored webcam
feed and then applied to Surrey's unmirrored Kinect images, which inverted
them: Surrey came out 99.4% left-slot while the project's own recordings and
the deployed browser pipeline are 74% right-slot.

A model trained on one convention reads the other's vectors as an empty hand
beside a populated one, which is what produced a 25% cross-dataset result
where letters scored either 0% or 100%.

Flipping the saved file is equivalent to re-extracting with the corrected
rule and takes seconds rather than half an hour. The leave-one-signer-out
accuracy will not change — a model learns whichever convention it is given,
as long as it is consistent — but everything downstream will now agree.

Also reports the per-class slot breakdown, which is worth seeing for a
dataset that mixes conventions.

    python scripts/data/fix_slots.py data/processed/surrey.npz
"""

import argparse
import collections
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402


def slot_of(vectors: np.ndarray) -> np.ndarray:
    """Per-vector: 'L', 'R', 'both' or 'none'."""
    left = np.abs(vectors[:, :63]).sum(axis=1) > 1e-6
    right = np.abs(vectors[:, 63:]).sum(axis=1) > 1e-6

    out = np.full(len(vectors), "none", dtype=object)
    out[left & ~right] = "L"
    out[right & ~left] = "R"
    out[left & right] = "both"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", help="Feature file to flip")
    ap.add_argument("--report-only", action="store_true",
                    help="Show the per-class breakdown and change nothing")
    args = ap.parse_args()

    path = pathlib.Path(args.npz)
    if not path.exists():
        print(f"No such file: {path}")
        return 1

    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    slots = slot_of(X)

    print(f"{path}  —  {len(X)} vectors\n")
    print(f"{'class':<8}{'L':>8}{'R':>8}{'both':>8}{'none':>8}")
    print("-" * 40)

    mixed = []
    for label in sorted(set(y.tolist())):
        mask = y == label
        counts = collections.Counter(slots[mask].tolist())
        print(
            f"{label:<8}{counts['L']:>8}{counts['R']:>8}"
            f"{counts['both']:>8}{counts['none']:>8}"
        )
        # A class split across both conventions is a contradictory signal.
        if counts["L"] and counts["R"]:
            minority = min(counts["L"], counts["R"])
            total = counts["L"] + counts["R"]
            if minority / total > 0.05:
                mixed.append((label, minority / total))

    print("-" * 40)
    overall = collections.Counter(slots.tolist())
    print(
        f"{'total':<8}{overall['L']:>8}{overall['R']:>8}"
        f"{overall['both']:>8}{overall['none']:>8}"
    )

    if mixed:
        print("\nClasses using both conventions:")
        for label, share in sorted(mixed, key=lambda kv: -kv[1]):
            print(f"  {label:<8}{share:>7.1%} in the minority slot")
        print("\nWithin one class that is a contradictory signal: the same")
        print("handshape appears in two different halves of the feature")
        print("vector, so the model must learn it twice.")

    if args.report_only:
        return 0

    backup = path.with_suffix(".npz.before-slot-fix")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"\nBacked up to {backup}")

    flipped = np.concatenate([X[:, 63:], X[:, :63]], axis=1)

    np.savez_compressed(
        path,
        X=flipped,
        y=y,
        signer=data["signer"],
        paths=data["paths"],
    )

    after = collections.Counter(slot_of(flipped).tolist())
    print(f"\nFlipped. Now L {after['L']}  R {after['R']}  "
          f"both {after['both']}  none {after['none']}")
    print(f"Wrote {path}")
    print("\nRetrain on the flipped file, then re-run the cross-dataset test.")

    return 0


if __name__ == "__main__":
    sys.exit(main())