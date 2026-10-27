"""
scripts/train/train_words.py — Word-level model: pretrain, ensemble, TTA

The final word-branch training. Four techniques, all aimed at the same
problem: 85 glosses with a median of 12 examples each.

1. VELOCITY FEATURES
   Frame-to-frame differences appended to each frame. A recurrent model can
   infer motion from positions, but with 12 examples per class it has no
   budget to learn what it can simply be told. Doubles the feature width.

2. PRETRAIN, THEN FINE-TUNE
   85 glosses clear the minimum example count; the other 15 do not, and
   their 129 sequences are currently discarded. They are still perfectly
   good for learning what signing looks like. The recurrent body is trained
   on all 100 glosses, then the head is replaced and the body fine-tuned on
   the 85. Pretraining uses only the fold's training signers, so the split
   stays disjoint.

3. ENSEMBLE
   Several models per fold with different initialisations, softmax averaged.
   Reliable on small data, and it shrinks the fold-to-fold spread, which
   matters here because that spread is what currently makes differences
   unreadable.

4. TEST-TIME AUGMENTATION
   Each test sequence is predicted several times under the same
   meaning-preserving transforms used in training, and the probabilities
   averaged. No labels are involved, so there is no leak.

Expect roughly 45-55% on 85 glosses. Published pose-based results on
WLASL100 sit at 50-65% using about twice this data, so that range is
competitive rather than disappointing. 80% on 85 classes is not reachable
from 1,204 sequences and should not be promised.

    python scripts/train/train_words.py data/processed/wlasl100_full.npz \
        --out models/words
"""

import argparse
import collections
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

HAND_DIM = 126


# ── Features ───────────────────────────────────────────────────────────

def add_velocity(X: np.ndarray) -> np.ndarray:
    """
    Append frame-to-frame differences to every frame.

    The first frame's velocity is zero, there being nothing before it.
    Shape goes from (n, steps, d) to (n, steps, 2d).
    """
    deltas = np.zeros_like(X)
    deltas[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    return np.concatenate([X, deltas], axis=2)


# ── Augmentation ───────────────────────────────────────────────────────
# Every transform preserves what the sign means. Mirroring is excluded: it
# swaps dominant and non-dominant hand, which changes or destroys meaning.

def time_warp(seq, rng, amount=0.2):
    steps = len(seq)
    source = np.clip(np.arange(steps) * (1.0 + rng.uniform(-amount, amount)),
                     0, steps - 1)
    out = np.empty_like(seq)
    for i, pos in enumerate(source):
        low, high = int(np.floor(pos)), min(int(np.ceil(pos)), steps - 1)
        frac = pos - low
        out[i] = seq[low] * (1 - frac) + seq[high] * frac
    return out


def jitter_location(seq, rng, amount=0.08, base_dim=132):
    """Shift body-frame hand positions. Hand shape is already translation-free."""
    if seq.shape[1] < base_dim:
        return seq
    out = seq.copy()
    offset = rng.uniform(-amount, amount, 3).astype(np.float32)
    for hand in range(2):
        block = slice(HAND_DIM + hand * 3, HAND_DIM + (hand + 1) * 3)
        present = np.abs(out[:, block]).sum(axis=1) > 1e-6
        out[present, block] += offset
    return out


def rotate_xy(seq, rng, degrees=10.0):
    angle = np.deg2rad(rng.uniform(-degrees, degrees))
    cos, sin = np.cos(angle), np.sin(angle)
    out = seq.copy()
    for start in range(0, seq.shape[1] - 2, 3):
        x = out[:, start].copy()
        y = out[:, start + 1].copy()
        out[:, start] = x * cos - y * sin
        out[:, start + 1] = x * sin + y * cos
    return out


def add_noise(seq, rng, sigma=0.01):
    return seq + rng.normal(0, sigma, seq.shape).astype(np.float32)


def transform(seq, rng, base_dim):
    """One random meaning-preserving transform chain."""
    out = seq
    if rng.random() < 0.7:
        out = time_warp(out, rng)
    if rng.random() < 0.7:
        out = jitter_location(out, rng, base_dim=base_dim)
    if rng.random() < 0.5:
        out = rotate_xy(out, rng)
    if rng.random() < 0.7:
        out = add_noise(out, rng)
    return out


def augment(X, y, factor, rng, base_dim):
    if factor <= 0:
        return X, y
    extra_X, extra_y = [], []
    for seq, label in zip(X, y):
        for _ in range(factor):
            extra_X.append(transform(seq, rng, base_dim))
            extra_y.append(label)
    return (np.concatenate([X, np.array(extra_X, dtype=np.float32)]),
            np.concatenate([y, np.array(extra_y)]))


# ── Model ──────────────────────────────────────────────────────────────

def build_body(steps, features):
    """
    The recurrent trunk, shared between pretraining and fine-tuning.

    Returned as its own model so the head can be swapped while the learned
    weights stay. That is what carries knowledge from the 100-gloss
    pretraining into the 85-gloss task.
    """
    from tensorflow import keras

    return keras.Sequential([
        keras.layers.Input(shape=(steps, features)),
        keras.layers.LSTM(128, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(64),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
    ], name="body")


def attach_head(body, classes):
    from tensorflow import keras

    return keras.Sequential(
        [body, keras.layers.Dense(classes, activation="softmax")]
    )


def fit(model, X_train, y_train, X_val, y_val, epochs, lr, patience=15):
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=32, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )],
    )
    return model


def predict_tta(model, X, rng, passes, base_dim):
    """Average softmax over the original plus `passes` transformed copies."""
    total = model.predict(X, verbose=0)
    for _ in range(passes):
        warped = np.array([transform(s, rng, base_dim) for s in X],
                          dtype=np.float32)
        total = total + model.predict(warped, verbose=0)
    return total / (1 + passes)


# ── Folds ──────────────────────────────────────────────────────────────

def group_signers(signer, folds):
    counts = collections.Counter(signer.tolist())
    load = {i: 0 for i in range(folds)}
    assignment = {}
    for s in sorted(counts, key=lambda s: -counts[s]):
        pick = min(load, key=lambda i: load[i])
        assignment[s] = pick
        load[pick] += counts[s]
    return assignment


def train_one(X_tr, y_tr, X_te, y_te, classes, args, rng, base_dim,
              pre=None):
    """
    One model: optional pretraining, then the real task.

    `pre` is (X, y, n_classes) for the pretraining stage, drawn from the
    same training signers so the fold stays disjoint.
    """
    body = build_body(X_tr.shape[1], X_tr.shape[2])

    if pre is not None:
        X_pre, y_pre, pre_classes = pre
        cut = rng.random(len(X_pre)) < 0.9
        fit(attach_head(body, pre_classes),
            X_pre[cut], y_pre[cut], X_pre[~cut], y_pre[~cut],
            args.pretrain_epochs, 1e-3, patience=10)

    model = attach_head(body, classes)
    # A lower rate after pretraining, so fine-tuning adapts the trunk
    # rather than overwriting what it learned.
    lr = 3e-4 if pre is not None else 1e-3
    return fit(model, X_tr, y_tr, X_te, y_te, args.epochs, lr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("--out", default="models/words")
    ap.add_argument("--min-per-class", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--pretrain-epochs", type=int, default=80)
    ap.add_argument("--augment", type=int, default=4)
    ap.add_argument("--ensemble", type=int, default=3,
                    help="Models averaged per fold (default 3)")
    ap.add_argument("--tta", type=int, default=4,
                    help="Extra augmented passes at test time (default 4)")
    ap.add_argument("--no-velocity", action="store_true")
    ap.add_argument("--no-pretrain", action="store_true")
    args = ap.parse_args()

    data = np.load(pathlib.Path(args.npz), allow_pickle=True)
    X_all = data["X"].astype(np.float32)
    y_all = np.array([str(v) for v in data["y"]])
    s_all = np.array([str(v) for v in data["signer"]])

    base_dim = X_all.shape[2]

    counts = collections.Counter(y_all.tolist())
    kept = sorted(g for g, n in counts.items() if n >= args.min_per_class)
    dropped = sorted(g for g, n in counts.items() if n < args.min_per_class)

    print(f"{len(X_all)} sequences · {len(counts)} glosses · "
          f"{len(set(s_all.tolist()))} signers · {base_dim} dims")

    if not kept:
        print(f"No gloss reaches {args.min_per_class} sequences.")
        return 1

    if not args.no_velocity:
        X_all = add_velocity(X_all)
        print(f"Velocity features appended: {base_dim} -> {X_all.shape[2]} dims")

    # The task: glosses with enough examples to learn and to score.
    mask = np.isin(y_all, kept)
    X, signer = X_all[mask], s_all[mask]
    index = {g: i for i, g in enumerate(kept)}
    y = np.array([index[v] for v in y_all[mask]], dtype=np.int64)

    # Pretraining pool: every gloss, including the thin ones.
    all_glosses = sorted(counts)
    pre_index = {g: i for i, g in enumerate(all_glosses)}
    y_pre_all = np.array([pre_index[v] for v in y_all], dtype=np.int64)

    print(f"\nTask: {len(kept)} glosses, {len(X)} sequences "
          f"(dropped {len(dropped)} below {args.min_per_class})")
    per_class = [counts[g] for g in kept]
    print(f"Per gloss: min {min(per_class)}, median "
          f"{int(np.median(per_class))}, max {max(per_class)}")

    if not args.no_pretrain:
        print(f"Pretraining pool: {len(all_glosses)} glosses, "
              f"{len(X_all)} sequences")

    print(f"Augmentation {args.augment}x · ensemble {args.ensemble} · "
          f"TTA {args.tta} passes")

    assignment = group_signers(signer, args.folds)
    sizes = collections.Counter(assignment[s] for s in signer)
    print(f"{args.folds} signer-disjoint folds, test sizes: "
          f"{[sizes[i] for i in range(args.folds)]}")

    total_models = args.folds * args.ensemble
    print(f"\nTraining {total_models} models"
          f"{' with pretraining' if not args.no_pretrain else ''}. "
          f"This takes a while.\n")

    accuracies = []
    single_accuracies = []
    started = time.time()

    for fold in range(args.folds):
        test = np.array([assignment[s] == fold for s in signer])
        if test.sum() == 0 or (~test).sum() == 0:
            continue

        # Pretraining must not see a test signer. The fold's test signers
        # are excluded from the pool as well as from the task data.
        test_signers = {s for s in signer[test]}
        pre_mask = np.array([s not in test_signers for s in s_all])

        probabilities = np.zeros((int(test.sum()), len(kept)), dtype=np.float32)

        for seed in range(args.ensemble):
            rng = np.random.default_rng(1000 * fold + seed)

            X_tr, y_tr = augment(X[~test], y[~test], args.augment, rng, base_dim)

            pre = None
            if not args.no_pretrain:
                pre = (X_all[pre_mask], y_pre_all[pre_mask], len(all_glosses))

            model = train_one(
                X_tr, y_tr, X[test], y[test], len(kept),
                args, rng, base_dim, pre,
            )

            p = predict_tta(model, X[test], rng, args.tta, base_dim)
            probabilities += p

            if seed == 0:
                single = float((p.argmax(axis=1) == y[test]).mean())
                single_accuracies.append(single)

        accuracy = float((probabilities.argmax(axis=1) == y[test]).mean())
        accuracies.append(accuracy)

        elapsed = (time.time() - started) / 60
        print(f"  fold {fold}: {accuracy:.2%}  "
              f"(single model {single_accuracies[-1]:.2%})  "
              f"[{elapsed:.0f} min elapsed]")

    arr = np.array(accuracies)
    single = np.array(single_accuracies)
    chance = 1.0 / len(kept)

    print(f"\n{'ensemble':<14}{arr.mean():>8.2%} ± {arr.std():.2%}")
    print(f"{'single model':<14}{single.mean():>8.2%} ± {single.std():.2%}")
    print(f"{'chance':<14}{chance:>8.2%}")
    print(f"{'ratio':<14}{arr.mean() / chance:>8.1f}x")

    # The shipped model: same recipe, trained on everyone.
    print(f"\nTraining the shipped model on all signers…")

    rng = np.random.default_rng(7)
    cut = rng.random(len(X)) < 0.9
    X_tr, y_tr = augment(X[cut], y[cut], args.augment, rng, base_dim)

    pre = (None if args.no_pretrain
           else (X_all, y_pre_all, len(all_glosses)))

    final = train_one(X_tr, y_tr, X[~cut], y[~cut], len(kept),
                      args, rng, base_dim, pre)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    final.save(out / "words.h5")
    (out / "labels.json").write_text(json.dumps(kept, indent=2))
    (out / "evaluation.json").write_text(json.dumps({
        "protocol": f"{args.folds} signer-disjoint folds",
        "glosses": len(kept),
        "sequences": int(len(X)),
        "signers": len(set(signer.tolist())),
        "feature_dim": int(X.shape[2]),
        "velocity": not args.no_velocity,
        "pretrained": not args.no_pretrain,
        "ensemble": args.ensemble,
        "tta_passes": args.tta,
        "augment_factor": args.augment,
        "folds_ensemble": [float(a) for a in accuracies],
        "folds_single": [float(a) for a in single_accuracies],
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "single_mean": float(single.mean()),
        "chance": chance,
        "ratio_to_chance": float(arr.mean() / chance),
        "dropped_glosses": dropped,
    }, indent=2))

    print(f"\nWrote {out}/words.h5, labels.json, evaluation.json")
    print(f"\nReport: {arr.mean():.2%} ± {arr.std():.2%} on {len(kept)} "
          f"glosses, {args.folds} signer-disjoint folds "
          f"({arr.mean() / chance:.0f}x chance).")
    print("Always state the class count beside the accuracy — the figure is")
    print("not comparable across different vocabulary sizes.")

    return 0


if __name__ == "__main__":
    sys.exit(main())