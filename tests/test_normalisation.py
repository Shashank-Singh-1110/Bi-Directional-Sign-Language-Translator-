import numpy as np
import pytest

from signbridge.normalization import (
    EPSILON,
    FEATURE_DIM,
    HAND_DIM,
    LANDMARKS,
    RAW_DIM,
    build_features,
    empty_hand,
    normalize_hand,
    normalize_keypoints,
)

RNG = np.random.default_rng(1110)


def random_hand() -> np.ndarray:
    """A plausible hand: landmarks clustered in a small region of frame."""
    base = RNG.uniform(0.3, 0.7, size=3)
    offsets = RNG.normal(0, 0.05, size=(LANDMARKS, 3))
    return (base + offsets).astype(np.float32).flatten()


# ── Shape contract ─────────────────────────────────────────────────────
def test_hand_output_shape():
    assert normalize_hand(random_hand()).shape == (HAND_DIM,)


def test_keypoints_output_shape():
    raw = RNG.uniform(0, 1, size=RAW_DIM).astype(np.float32)
    assert normalize_keypoints(raw).shape == (FEATURE_DIM,)


@pytest.mark.parametrize("bad_size", [62, 64, 126, 0])
def test_wrong_hand_size_rejected(bad_size):
    with pytest.raises(ValueError):
        normalize_hand(np.zeros(bad_size, dtype=np.float32))


@pytest.mark.parametrize("bad_size", [126, 257, 259])
def test_wrong_raw_size_rejected(bad_size):
    with pytest.raises(ValueError):
        normalize_keypoints(np.zeros(bad_size, dtype=np.float32))


# ── Core invariants ────────────────────────────────────────────────────
def test_wrist_moves_to_origin():
    out = normalize_hand(random_hand())
    np.testing.assert_allclose(out[:3], [0, 0, 0], atol=1e-6)


def test_furthest_landmark_is_unit_distance():
    out = normalize_hand(random_hand()).reshape(LANDMARKS, 3)
    max_dist = np.linalg.norm(out, axis=1).max()
    assert max_dist == pytest.approx(1.0, abs=1e-5)


def test_no_landmark_exceeds_unit_distance():
    out = normalize_hand(random_hand()).reshape(LANDMARKS, 3)
    assert np.linalg.norm(out, axis=1).max() <= 1.0 + 1e-5


# ── The property the system is built on ────────────────────────────────
@pytest.mark.parametrize("scale", [0.25, 0.5, 2.0, 4.0])
def test_scale_invariance(scale):
    """
    The same hand shape at a different apparent size — i.e. the signer
    standing closer or further from the camera — must normalize to the
    same vector. This is the domain-gap fix.
    """
    hand = random_hand()
    scaled = (hand.reshape(LANDMARKS, 3) * scale).flatten()

    a = normalize_hand(hand)
    b = normalize_hand(scaled)

    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert cosine > 0.9999, f"Cosine similarity {cosine:.6f} at scale {scale}"


@pytest.mark.parametrize("shift", [-0.2, 0.1, 0.3])
def test_translation_invariance(shift):
    """Signing in a different part of the frame must not change the vector."""
    hand = random_hand()
    moved = (hand.reshape(LANDMARKS, 3) + shift).flatten()

    np.testing.assert_allclose(normalize_hand(hand), normalize_hand(moved), atol=1e-5)


# ── Absent-hand path ───────────────────────────────────────────────────
def test_empty_hand_stays_zero():
    """The epsilon guard must prevent a divide-by-zero NaN."""
    out = normalize_hand(empty_hand())
    assert np.all(out == 0)
    assert not np.any(np.isnan(out))


def test_single_handed_sign_keeps_other_half_zero():
    """Half the vector empty is the normal case, not an error case."""
    features = build_features(empty_hand(), random_hand())

    assert np.all(features[:HAND_DIM] == 0)
    assert np.any(features[HAND_DIM:] != 0)
    assert not np.any(np.isnan(features))


def test_near_degenerate_hand_does_not_explode():
    """All landmarks almost coincident — below epsilon, leave it alone."""
    hand = np.full(HAND_DIM, 0.5, dtype=np.float32)
    hand[3:] += EPSILON / 10

    out = normalize_hand(hand)
    assert not np.any(np.isnan(out))
    assert np.all(np.abs(out) < 1.0 + 1e-5)


# ── Hand ordering ──────────────────────────────────────────────────────
def test_hand_order_is_not_symmetric():
    """
    Swapping hands must produce a different vector. If this ever passes
    trivially, the two halves are being mixed somewhere.
    """
    left, right = random_hand(), random_hand()

    a = build_features(left, right)
    b = build_features(right, left)

    assert not np.allclose(a, b)


def test_build_features_matches_slicing():
    """build_features and normalize_keypoints must agree."""
    raw = RNG.uniform(0, 1, size=RAW_DIM).astype(np.float32)

    np.testing.assert_allclose(
        normalize_keypoints(raw),
        build_features(raw[132:195], raw[195:258]),
        atol=1e-7,
    )


# ── Regression guard against the original implementation ───────────────
def test_matches_reference_implementation():
    """
    The original in-line implementation from normalize_and_retrain.py,
    reproduced here. The extracted module must not have changed behaviour.
    """

    def reference(hand_63):
        pts = hand_63.reshape(21, 3)
        wrist = pts[0].copy()
        pts = pts - wrist
        dists = np.linalg.norm(pts, axis=1)
        max_d = dists.max()
        if max_d > 1e-6:
            pts = pts / max_d
        return pts.flatten()

    for _ in range(20):
        hand = random_hand()
        np.testing.assert_allclose(normalize_hand(hand), reference(hand), atol=1e-6)
