# Wrist-Origin Normalization — Solving the Domain Gap

The single most impactful technical decision in this project.

---

## The Problem

Every sign language ML system that uses raw landmark coordinates suffers from the same deployment failure: **domain gap**.

You train with the signer 50cm from the camera. You deploy at 100cm. The raw pixel coordinates are completely different numbers for the same hand shape. Accuracy collapses.

This is not a minor inconvenience. Zhang and Jiang (2024), in their survey of 200+ sign language recognition papers, identified domain gap as the **primary unsolved deployment challenge** in the field. Most papers recommend solving it with data augmentation — recording at multiple distances, lighting conditions, and positions.

We solved it with a 10-line normalization function instead.

---

## The Solution

### Step 1 — Translation to Wrist Origin

MediaPipe gives us 21 landmarks per hand, each with (x, y, z) coordinates in camera space:

```
Landmark 0 (wrist):     (0.42, 0.71, 0.003)
Landmark 1 (thumb base): (0.45, 0.68, 0.012)
...
```

We subtract the wrist position from all 21 landmarks:

```python
wrist = pts[0].copy()
pts   = pts - wrist
```

Now every coordinate is relative to the wrist, not the camera:

```
Landmark 0 (wrist):     (0.00, 0.00, 0.000)  ← always zero
Landmark 1 (thumb base): (0.03, -0.03, 0.009) ← relative position
...
```

**What this buys:** The absolute position of your hand in the frame no longer matters. Top-left, bottom-right, centre — same output.

### Step 2 — Scale Normalization

Even after translation, a hand close to the camera spans more pixels than a hand further away. The relative positions are the same, but the magnitudes differ.

We divide by the maximum Euclidean distance from the wrist to any landmark:

```python
max_d = np.linalg.norm(pts, axis=1).max()
if max_d > 1e-6:
    pts = pts / max_d
```

**What this buys:** The entire hand now fits within a unit sphere. Distance from the camera is irrelevant.

### Full Function

```python
def normalize_hand(hand_63):
    pts   = hand_63.reshape(21, 3)
    wrist = pts[0].copy()
    pts   = pts - wrist
    max_d = np.linalg.norm(pts, axis=1).max()
    if max_d > 1e-6:
        pts = pts / max_d
    return pts.flatten()  # 63-dim normalized vector per hand
```

Applied to both hands → 126-dim normalized feature vector per frame.

---

## Verification

We verified the normalization empirically:

| Distance | Cosine Similarity to 30cm reference |
|----------|-------------------------------------|
| 30cm     | 1.000 (baseline) |
| 60cm     | 0.997 |
| 100cm    | 0.993 |

The same hand configuration at different distances produces vectors with >0.99 cosine similarity.

---

## What We Dropped

The original MediaPipe Holistic vector is 1,662 dimensions:
- 468 face landmarks × 3 = 1,404 dims
- 33 pose landmarks × 4 = 132 dims
- 21 hand landmarks × 3 × 2 = 126 dims

We keep only the 126 hand dimensions. **Face and pose landmarks make things worse.**

Khartheesvar et al. (2024) found that face landmarks introduced confusion between signs performed near the face. Pose landmarks create false correlations when the signer's body position shifts. Removing them:
- Reduced dimensionality by 92.4%
- Resolved I LOVE YOU / Thanks confusion
- Eliminated pose-induced domain gap entirely
- Improved accuracy

Less data, better results. The 126-dim hand-only normalized vector is the right representation for this problem.

---

## Impact on Training

With normalization, the model learns hand *shapes* and *configurations* — not absolute positions or scales. A sign made by a tall person with large hands produces the same normalized vector as the same sign made by someone with small hands close to the camera.

This is why we could train on a single-signer personal dataset and have it generalize to other signers — the normalization removes the signer-specific scale and position biases from the training data.
