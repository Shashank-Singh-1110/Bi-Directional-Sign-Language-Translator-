"""
signbridge/normalization.py — Wrist-origin hand normalization

This is the single source of truth for the transform: the training pipeline,
the Flask backend and the browser all have to agree with what is here.

The transform is what makes the model distance-invariant — the same sign at
30 cm and 100 cm yields vectors with cosine similarity above 0.99.
"""

import numpy as np

LANDMARKS = 21
HAND_DIM = LANDMARKS * 3  # 63
FEATURE_DIM = HAND_DIM * 2  # 126
RAW_DIM = 258  # pose(132) + left(63) + right(63)

EPSILON = 1e-6

# Slice boundaries in the raw 258-dim MediaPipe Holistic vector
LEFT_SLICE = slice(132, 195)
RIGHT_SLICE = slice(195, 258)


def normalize_hand(hand_63: np.ndarray) -> np.ndarray:
    """
    Normalize one hand's 63 values (21 landmarks x xyz).

    1. Translate so landmark 0 (the wrist) sits at the origin
    2. Divide by the largest Euclidean distance from the wrist

    An all-zero input — meaning the hand was not detected — returns all
    zeros rather than NaN. Most single-handed signs rely on this, since
    half the feature vector is empty for them.
    """
    hand_63 = np.asarray(hand_63, dtype=np.float32)
    if hand_63.shape != (HAND_DIM,):
        raise ValueError(f"Expected shape ({HAND_DIM},), got {hand_63.shape}")

    pts = hand_63.reshape(LANDMARKS, 3) - hand_63.reshape(LANDMARKS, 3)[0]

    max_d = np.linalg.norm(pts, axis=1).max()
    if max_d > EPSILON:
        pts = pts / max_d

    return pts.flatten()


def normalize_keypoints(raw_258: np.ndarray) -> np.ndarray:
    """
    Convert a raw 258-dim Holistic vector to the 126-dim model input.

    Layout in:  [pose(132), left_hand(63), right_hand(63)]
    Layout out: [left_hand(63), right_hand(63)]

    The pose block is discarded — it carried no signal the hands did not
    already provide, and dropping it cut the input dimensionality by 92%.
    """
    raw_258 = np.asarray(raw_258, dtype=np.float32)
    if raw_258.shape != (RAW_DIM,):
        raise ValueError(f"Expected shape ({RAW_DIM},), got {raw_258.shape}")

    return np.concatenate(
        [
            normalize_hand(raw_258[LEFT_SLICE]),
            normalize_hand(raw_258[RIGHT_SLICE]),
        ]
    )


def build_features(left_63: np.ndarray, right_63: np.ndarray) -> np.ndarray:
    """
    Build the model input from two hands directly.

    Pass np.zeros(63) for a hand that was not detected. Hand order is
    [left, right] and is not interchangeable.
    """
    return np.concatenate([normalize_hand(left_63), normalize_hand(right_63)])


def empty_hand() -> np.ndarray:
    """A not-detected hand."""
    return np.zeros(HAND_DIM, dtype=np.float32)


# ── Canonical hand placement ────────────────────────────────────────────
#
# For a one-handed sign only one block is populated, and which one depends
# on the hand the signer used. A model trained mostly on one block has
# nothing to match when the features arrive in the other, so it fails
# completely rather than degrading.
#
# This was visible three ways before it was understood:
#
#   - the deployed app recognised one hand and ignored the other
#   - the recorded dataset splits 74% right-slot, 25% left-slot, per letter
#   - cross-dataset accuracy sat at 25-38%, with a different quarter of the
#     letters failing depending on which convention was aligned
#
# Canonicalising removes the ambiguity: a one-handed sign always occupies
# the same block, so the model learns the handshape once rather than twice,
# and a signer may use either hand.
#
# Two-handed signs are left alone — there the two blocks carry genuinely
# different information and collapsing them would destroy it.

CANONICAL_SLOT = "right"  # the majority convention in the recorded data


def hand_present(hand_63: np.ndarray) -> bool:
    """Whether a hand block carries data rather than being all zeros."""
    return bool(np.abs(hand_63).sum() > EPSILON)


def canonicalise(features_126: np.ndarray) -> np.ndarray:
    """
    Place a lone hand in a fixed block, leaving two-handed signs untouched.

    Returns a new array; the input is not modified.
    """
    features = np.asarray(features_126, dtype=np.float32).copy()

    # Copied, not sliced: a view into `features` would be zeroed by the
    # assignment below before it could be written to the other block.
    left = features[:HAND_DIM].copy()
    right = features[HAND_DIM:]

    if hand_present(left) and not hand_present(right):
        features[:HAND_DIM] = 0.0
        features[HAND_DIM:] = left

    return features


def canonicalise_batch(X: np.ndarray) -> np.ndarray:
    """
    Canonicalise a stack of feature vectors.

    Vectorised rather than looped: these run over tens of thousands of rows
    during extraction.
    """
    X = np.asarray(X, dtype=np.float32).copy()

    has_left = np.abs(X[:, :HAND_DIM]).sum(axis=1) > EPSILON
    has_right = np.abs(X[:, HAND_DIM:]).sum(axis=1) > EPSILON

    lone_left = has_left & ~has_right

    X[lone_left, HAND_DIM:] = X[lone_left, :HAND_DIM]
    X[lone_left, :HAND_DIM] = 0.0

    return X
