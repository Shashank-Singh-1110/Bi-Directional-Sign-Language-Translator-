"""
augment_and_retrain.py — Train on synthetically varied data

The recorded dataset has almost no variation: two recordings of the same
sign score 0.999 cosine similarity. The model trained on it memorizes
templates, which is why it reports 100% on held-out sequences and fails on
anything from a different camera or hand (Android capture scored 0.775).

This generates variation the recording session never captured, trains on it,
and evaluates against a grouped holdout so the reported number means
something.

Nothing is re-recorded. Augmentation cannot invent a different person's hand
proportions, but it directly targets the failure that was measured.

Run: python scripts/train/augment_and_retrain.py
"""

import json
import os

import numpy as np

DATA_PATH   = 'data/DATASET_NORM'
MODEL_OUT   = 'models/action_aug.h5'
REPORT_DIR  = 'documents/evaluation'

SEQ_LENGTH  = 30
FEATURE_DIM = 126
HAND_DIM    = 63
LANDMARKS   = 21

# Augmentations per original sequence. 960 x 20 = ~19,200 sequences.
VARIANTS = 20

# Hold out whole recording groups, not random sequences — consecutive
# recordings are near-duplicates, so a random split leaks.
HOLDOUT_FROM = 24

EPOCHS      = 150
BATCH_SIZE  = 32
PATIENCE    = 20
SEED        = 42

ACTIONS = [
    'Hello', 'Thanks', 'Yes', 'I LOVE YOU', 'No', 'Sorry',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

rng = np.random.default_rng(SEED)


# ── Augmentations ──────────────────────────────────────────────────────
# All operate on normalized 126-dim frames and renormalize afterwards, so
# the wrist-at-origin, unit-max-distance invariant is preserved.

def renormalize(hand):
    """Wrist to origin, divide by furthest landmark. Zeros stay zeros."""
    if not np.any(hand):
        return hand

    pts = hand.reshape(LANDMARKS, 3)
    pts = pts - pts[0]

    max_d = np.linalg.norm(pts, axis=1).max()
    if max_d > 1e-6:
        pts = pts / max_d

    return pts.flatten()


def rotate(hand, degrees):
    """In-plane rotation. Covers hand tilt and camera angle."""
    if not np.any(hand):
        return hand

    rad = np.radians(degrees)
    c, s = np.cos(rad), np.sin(rad)

    pts = hand.reshape(LANDMARKS, 3).copy()
    x, y = pts[:, 0].copy(), pts[:, 1].copy()
    pts[:, 0] = c * x - s * y
    pts[:, 1] = s * x + c * y

    return pts.flatten()


def jitter(hand, sigma):
    """
    Per-landmark noise. This is the important one: it covers the difference
    between MediaPipe Holistic (which produced the training data) and
    HandLandmarker (which Android runs), as well as ordinary tracking noise.
    """
    if not np.any(hand):
        return hand

    pts = hand.reshape(LANDMARKS, 3).copy()
    pts += rng.normal(0, sigma, pts.shape)
    pts[0] = 0.0  # keep the wrist at the origin
    return pts.flatten()


def shear(hand, kx, ky):
    """
    Slight anisotropic stretch. Covers aspect-ratio differences between
    cameras, which wrist normalization does NOT remove.
    """
    if not np.any(hand):
        return hand

    pts = hand.reshape(LANDMARKS, 3).copy()
    pts[:, 0] *= kx
    pts[:, 1] *= ky
    return pts.flatten()


def augment_frame(frame, params):
    """Apply one parameter set to both hands of a frame."""
    left, right = frame[:HAND_DIM], frame[HAND_DIM:]

    out = []
    for hand in (left, right):
        h = rotate(hand, params['rot'])
        h = shear(h, params['kx'], params['ky'])
        h = jitter(h, params['sigma'])
        out.append(renormalize(h))

    result = np.concatenate(out)

    if params['mirror']:
        # Horizontal flip plus slot swap — turns a right-handed recording
        # into a left-handed one. Also fixes the dataset's inconsistency
        # about which hand recorded which sign.
        pts = result.reshape(2 * LANDMARKS, 3).copy()
        pts[:, 0] *= -1
        flipped = pts.flatten()
        result = np.concatenate([flipped[HAND_DIM:], flipped[:HAND_DIM]])

    return result.astype(np.float32)


def augment_sequence(seq):
    """One sequence -> one augmented sequence, consistent across frames."""
    params = {
        'rot':    rng.uniform(-25, 25),
        'kx':     rng.uniform(0.85, 1.15),
        'ky':     rng.uniform(0.85, 1.15),
        'sigma':  rng.uniform(0.005, 0.025),
        'mirror': rng.random() < 0.5,
    }

    frames = [augment_frame(f, params) for f in seq]

    # Temporal jitter: drop or duplicate a few frames, then pad back to 30.
    if rng.random() < 0.4:
        keep = sorted(rng.choice(SEQ_LENGTH, size=SEQ_LENGTH - 3, replace=False))
        frames = [frames[i] for i in keep]
        while len(frames) < SEQ_LENGTH:
            frames.append(frames[-1])

    return np.array(frames[:SEQ_LENGTH], dtype=np.float32)


# ── Data ───────────────────────────────────────────────────────────────
def load_with_groups():
    X, y, groups = [], [], []

    for label, action in enumerate(ACTIONS):
        a_dir = os.path.join(DATA_PATH, action)
        if not os.path.isdir(a_dir):
            print(f"[WARN] missing: {action}")
            continue

        for seq in sorted(os.listdir(a_dir)):
            if not seq.isdigit():
                continue
            s_dir = os.path.join(a_dir, seq)
            if not os.path.isdir(s_dir):
                continue

            frames = []
            for f in range(SEQ_LENGTH):
                fp = os.path.join(s_dir, f'{f}.npy')
                if not os.path.exists(fp):
                    break
                frames.append(np.load(fp))

            if len(frames) == SEQ_LENGTH:
                X.append(frames)
                y.append(label)
                groups.append(int(seq))

    return (np.array(X, dtype=np.float32),
            np.array(y),
            np.array(groups))


def main():
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("=" * 70)
    print("  Augmented retraining")
    print("=" * 70)

    # ---- Load ----
    X, y, groups = load_with_groups()
    print(f"\n[DATA] {len(X)} original sequences, {len(set(y))} classes")

    # ---- Grouped split BEFORE augmenting ----
    # Augmenting first would leak: variants of the same recording would
    # land in both train and test.
    test_mask = groups >= HOLDOUT_FROM
    X_tr_raw, y_tr_raw = X[~test_mask], y[~test_mask]
    X_te_raw, y_te_raw = X[test_mask], y[test_mask]

    print(f"[SPLIT] train {len(X_tr_raw)} (seq < {HOLDOUT_FROM})")
    print(f"[SPLIT] test  {len(X_te_raw)} (seq >= {HOLDOUT_FROM})")

    # ---- Augment the training half only ----
    print(f"\n[AUG] Generating {VARIANTS} variants per sequence...")
    X_aug, y_aug = [], []

    for i, (seq, label) in enumerate(zip(X_tr_raw, y_tr_raw)):
        X_aug.append(seq)          # keep the original
        y_aug.append(label)
        for _ in range(VARIANTS - 1):
            X_aug.append(augment_sequence(seq))
            y_aug.append(label)

        if (i + 1) % 100 == 0:
            print(f"      {i + 1}/{len(X_tr_raw)}")

    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug)
    print(f"[AUG] {len(X_aug)} training sequences")

    # ---- Did augmentation actually create variation? ----
    print("\n[CHECK] Intra-class similarity after augmentation")
    for action in ('I LOVE YOU', 'Hello'):
        if action not in ACTIONS:
            continue
        idx = np.where(y_aug == ACTIONS.index(action))[0][:40]
        if len(idx) < 2:
            continue
        mids = X_aug[idx][:, 15, :]
        sims = []
        for a in range(len(mids)):
            for b in range(a + 1, len(mids)):
                na, nb = np.linalg.norm(mids[a]), np.linalg.norm(mids[b])
                if na > 1e-9 and nb > 1e-9:
                    sims.append(np.dot(mids[a], mids[b]) / (na * nb))
        if sims:
            print(f"        {action:<12} mean {np.mean(sims):.3f}  "
                  f"min {np.min(sims):.3f}   (was ~0.999)")

    # ---- Train ----
    yc_train = to_categorical(y_aug, num_classes=len(ACTIONS))
    yc_test = to_categorical(y_te_raw, num_classes=len(ACTIONS))

    model = Sequential([
        Input(shape=(SEQ_LENGTH, FEATURE_DIM)),
        LSTM(128, return_sequences=True, activation='tanh'),
        LSTM(64, return_sequences=False, activation='tanh'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(ACTIONS), activation='softmax'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'],
    )

    print(f"\n[TRAIN] {model.count_params():,} parameters")

    model.fit(
        X_aug, yc_train,
        validation_data=(X_te_raw, yc_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=PATIENCE,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=8, min_lr=1e-6),
        ],
        verbose=1,
    )

    model.save(MODEL_OUT)

    # ---- Evaluate ----
    probs = model.predict(X_te_raw, verbose=0)
    preds = np.argmax(probs, axis=1)
    acc = (preds == y_te_raw).mean() * 100
    conf = probs.max(axis=1) * 100
    correct = preds == y_te_raw

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Held-out accuracy   {acc:.2f}%  "
          f"({correct.sum()}/{len(y_te_raw)})")
    print(f"  Mean confidence     {conf[correct].mean():.1f}% when correct")
    if (~correct).any():
        print(f"                      {conf[~correct].mean():.1f}% when wrong")
    print(f"  Lowest correct      {conf[correct].min():.1f}%")

    print(f"\n  Model written to {MODEL_OUT}")

    with open(os.path.join(REPORT_DIR, 'augmented_metrics.json'), 'w') as fh:
        json.dump({
            'model': MODEL_OUT,
            'variants_per_sequence': VARIANTS,
            'training_sequences': int(len(X_aug)),
            'test_sequences': int(len(X_te_raw)),
            'holdout_from_sequence': HOLDOUT_FROM,
            'accuracy': round(float(acc), 2),
            'mean_confidence_correct': round(float(conf[correct].mean()), 2),
            'min_confidence_correct': round(float(conf[correct].min()), 2),
        }, fh, indent=2)

    print("\n" + "=" * 70)
    print("READING THE RESULT")
    print("=" * 70)
    print("  A number below the old 98.96% is expected and is the point.")
    print("  The old figure measured recall of near-identical sequences.")
    print("  This one measures recognition of varied input.")
    print()
    print("  Watch the confidence spread. Previously every prediction was")
    print("  100.0%, which is what memorization looks like. A healthy model")
    print("  shows a range.")
    print()
    print("  Next: convert with")
    print("    python scripts/convert/convert_native_tflite.py")
    print("  pointing KERAS_MODEL at", MODEL_OUT)
    print("  then copy the .tflite into android/app/src/main/assets/")


if __name__ == '__main__':
    main()