"""
scripts/train/measure_similarity.py — Is this dataset actually diverse?

The project's own recorded dataset measured 0.999 mean intra-class cosine
similarity: 30 sequences per sign, recorded back to back by one person, so
near-duplicates of each other. A train/test split over that data puts a copy
of nearly every training example into the test set, and the resulting 98.96%
measures template recall rather than generalisation.

Before training anything on a replacement dataset, measure the same quantity
on it. If it comes back near 1.0 as well, the replacement fixes nothing and
that is worth knowing in an hour rather than a week.

Two numbers matter, and the gap between them is the real signal:

  within-signer   how alike two images of the same letter by the same person
  across-signer   how alike two images of the same letter by different people

A dataset where both are ~1.0 has no diversity to learn from. A dataset
where across-signer is clearly lower is one where a model has to learn the
handshape rather than memorise a template.

    python scripts/train/measure_similarity.py data/processed/surrey.npz
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

RECORDED_BASELINE = 0.999   # measured on the project's own dataset


def mean_pairwise_cosine(vectors: np.ndarray, cap: int, rng) -> float:
    """
    Mean cosine similarity over random pairs.

    Sampled rather than exhaustive: a class with 2,500 vectors has over
    three million pairs, and a few thousand of them estimate the mean to
    more precision than the question needs.
    """
    if len(vectors) < 2:
        return float("nan")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.maximum(norms, 1e-9)

    i = rng.integers(0, len(unit), cap)
    j = rng.integers(0, len(unit), cap)
    keep = i != j
    if not keep.any():
        return float("nan")

    return float(np.sum(unit[i[keep]] * unit[j[keep]], axis=1).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", help="Feature file from extract_landmarks.py")
    ap.add_argument("--pairs", type=int, default=4000,
                    help="Random pairs sampled per measurement")
    args = ap.parse_args()

    path = pathlib.Path(args.npz)
    if not path.exists():
        print(f"No such file: {path}")
        return 1

    data = np.load(path, allow_pickle=True)
    X, y, signer = data["X"], data["y"], data["signer"]

    classes = sorted(set(y.tolist()))
    signers = sorted(set(signer.tolist()))

    print(f"{len(X)} vectors · {len(classes)} classes · {len(signers)} signers\n")

    rng = np.random.default_rng(0)

    print(f"{'class':<8}{'n':>8}{'within-signer':>16}{'across-signer':>16}{'gap':>8}")
    print("-" * 56)

    within_all, across_all = [], []

    for label in classes:
        in_class = X[y == label]
        class_signers = signer[y == label]

        # Within: pairs drawn from one signer at a time, averaged.
        per_signer = []
        for s in signers:
            subset = in_class[class_signers == s]
            score = mean_pairwise_cosine(subset, args.pairs, rng)
            if not np.isnan(score):
                per_signer.append(score)

        within = float(np.mean(per_signer)) if per_signer else float("nan")

        # Across: pairs drawn from the whole class, which mixes signers.
        across = mean_pairwise_cosine(in_class, args.pairs, rng)

        within_all.append(within)
        across_all.append(across)

        print(
            f"{label:<8}{len(in_class):>8}{within:>16.4f}"
            f"{across:>16.4f}{within - across:>8.4f}"
        )

    print("-" * 56)

    within_mean = float(np.nanmean(within_all))
    across_mean = float(np.nanmean(across_all))

    print(f"{'mean':<8}{len(X):>8}{within_mean:>16.4f}{across_mean:>16.4f}"
          f"{within_mean - across_mean:>8.4f}")

    print(f"\nRecorded dataset, for comparison:   {RECORDED_BASELINE:.4f}")
    print(f"This dataset, across signers:       {across_mean:.4f}")

    print()
    if across_mean >= 0.99:
        print("VERDICT: near-duplicate, like the recorded set.")
        print("Training on this reproduces the same problem. A high accuracy")
        print("here would again be template recall, not generalisation.")
    elif across_mean >= 0.95:
        print("VERDICT: some diversity, but limited.")
        print("Report the accuracy with this number beside it — the two")
        print("belong together.")
    else:
        print("VERDICT: genuinely diverse.")
        print("Same-letter examples differ substantially across signers, so")
        print("a model has to learn the handshape rather than memorise a")
        print("template. Accuracy measured here means what it claims.")

    if within_mean - across_mean > 0.02:
        print(f"\nWithin-signer similarity is {within_mean - across_mean:.4f}")
        print("higher than across-signer. That gap is signer identity —")
        print("exactly what a signer-disjoint split has to hold out, and")
        print("the justification for not using a random split.")

    return 0


if __name__ == "__main__":
    sys.exit(main())