"""
Addition to signbridge/normalization.py — canonical hand placement

Paste these two functions into signbridge/normalization.py. Nothing already
in that file changes.

The problem they solve:

A 126-dim vector is [left_hand(63), right_hand(63)]. For a one-handed sign
only one block is populated, and which one depends on which hand the signer
used. A model trained mostly on one block has nothing to match when the
features arrive in the other, so it fails completely rather than degrading.

This was visible three ways before it was understood:

  - the deployed app recognised one hand and ignored the other
  - the recorded dataset splits 74% right-slot, 25% left-slot, per letter
  - cross-dataset accuracy sat at 25-38% with a different quarter of the
    letters failing depending on which convention was aligned

Canonicalising removes the ambiguity: a one-handed sign always occupies the
same block, so the model learns the handshape once rather than twice, and
a signer may use either hand.

Two-handed signs are left alone. There the two blocks carry genuinely
different information and collapsing them would destroy it.
"""

import numpy as np

HAND_DIM = 63
EPSILON = 1e-6


def hand_present(hand_63: np.ndarray) -> bool:
    """Whether a hand block carries data rather than being all zeros."""
    return bool(np.abs(hand_63).sum() > EPSILON)


def canonicalise(features_126: np.ndarray) -> np.ndarray:
    """
    Place a lone hand in a fixed block, leaving two-handed signs untouched.

    The canonical slot is the right block — chosen because it is the
    majority convention in the project's own recordings (74%), so existing
    data needs the least movement.

    Returns a new array; the input is not modified.
    """
    features = np.asarray(features_126, dtype=np.float32).copy()

    # Copied, not sliced: a view into `features` would be zeroed by the
    # assignment below before it could be written to the other block.
    left = features[:HAND_DIM].copy()
    right = features[HAND_DIM:]

    # Only a lone left hand moves. Both-hands and right-only are already
    # canonical, and neither-hand has nothing to place.
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

    left = X[:, :HAND_DIM]
    right = X[:, HAND_DIM:]

    has_left = np.abs(left).sum(axis=1) > EPSILON
    has_right = np.abs(right).sum(axis=1) > EPSILON

    lone_left = has_left & ~has_right

    X[lone_left, HAND_DIM:] = X[lone_left, :HAND_DIM]
    X[lone_left, :HAND_DIM] = 0.0

    return X