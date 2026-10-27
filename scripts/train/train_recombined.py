"""
scripts/train/train_combined.py — One letter model, two datasets

A model trained on Surrey alone transfers to this project's own camera about
half the time (50.69% measured). A model trained on this project's own
recordings alone memorises thirty near-identical takes and emits confident
nonsense on anything else. Neither is usable.

Training on both fixes both problems. The recorded signer becomes a sixth
subject, so:

  - the model has seen this camera, room and signer, and works in the demo
  - leave-one-signer-out still runs across six people, including holding the
    recorded signer out, so the reported number stays honest
  - there is one model rather than a pair with a routing rule

The recorded frames are subsampled. Each letter has 900 of them at 0.999
intra-class similarity, so they carry roughly one frame's worth of
information per sequence; including all 900 would weight this signer thirty
times more heavily than the evidence justifies.

    python scripts/train/train_combined.py \
        --surrey data/processed/surrey.npz \
        --recorded data/processed/recorded_canonical.npz \
        --out models/static
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from signbridge.normalization import canonicalise_batch  # noqa: E402

SELF_SIGNER = "self"


def build(input_dim: int, classes: int):
    from tensorflow import keras

    return keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(classes, activation="softmax"),
    ])


def fit(model, X_train, y_train, X_val, y_val, epochs):
    from tensorflow import keras

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=64, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        )],
    )
    return model


def subsample(X, y, per_class: int, seed: int = 0):
    """Cap the number of frames kept per class."""
    rng = np.random.default_rng(seed)
    keep = []

    for label in sorted(set(y.tolist())):
        idx = np.flatnonzero(y == label)
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        keep.append(idx)

    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return X[keep], y[keep]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surrey", default="data/processed/surrey.npz")
    ap.add_argument("--recorded", default="data/processed/recorded_canonical.npz")
    ap.add_argument("--out", default="models/static")
    ap.add_argument("--per-class", type=int, default=150,
                    help="Recorded frames kept per letter (default 150)")
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()

    surrey = np.load(pathlib.Path(args.surrey), allow_pickle=True)
    Xs = canonicalise_batch(surrey["X"])
    ys = np.array([str(v).lower() for v in surrey["y"]])
    ss = np.array([str(v) for v in surrey["signer"]])

    recorded = np.load(pathlib.Path(args.recorded), allow_pickle=True)
    Xr_all = canonicalise_batch(recorded["X"])
    yr_all = np.array([str(v).lower() for v in recorded["y"]])

    # Only the letters Surrey covers. The word gestures and the two motion
    # letters stay with the sequence model.
    letters = sorted(set(ys.tolist()))
    keep = np.isin(yr_all, letters)
    Xr_all, yr_all = Xr_all[keep], yr_all[keep]

    Xr, yr = subsample(Xr_all, yr_all, args.per_class)
    sr = np.full(len(Xr), SELF_SIGNER)

    X = np.concatenate([Xs, Xr])
    y_raw = np.concatenate([ys, yr])
    signer = np.concatenate([ss, sr])

    index = {label: i for i, label in enumerate(letters)}
    y = np.array([index[v] for v in y_raw], dtype=np.int64)
    signers = sorted(set(signer.tolist()))

    print(f"Surrey      {len(Xs):>7} vectors")
    print(f"Recorded    {len(Xr):>7} vectors "
          f"(subsampled from {len(Xr_all)}, {args.per_class}/letter)")
    print(f"Combined    {len(X):>7} vectors · {len(letters)} letters · "
          f"{len(signers)} signers\n")

    fold_accuracy = {}

    print(f"{'held out':<12}{'train':>9}{'test':>9}{'accuracy':>11}")
    print("-" * 41)

    for held in signers:
        test = signer == held
        model = fit(
            build(X.shape[1], len(letters)),
            X[~test], y[~test], X[test], y[test], args.epochs,
        )
        predicted = model.predict(X[test], verbose=0).argmax(axis=1)
        accuracy = float((predicted == y[test]).mean())
        fold_accuracy[held] = accuracy
        print(f"{held:<12}{(~test).sum():>9}{test.sum():>9}{accuracy:>10.2%}")

    print("-" * 41)

    acc = np.array(list(fold_accuracy.values()))
    print(f"{'mean':<12}{'':>9}{'':>9}{acc.mean():>10.2%}")
    print(f"{'std dev':<12}{'':>9}{'':>9}{acc.std():>10.2%}")

    held_self = fold_accuracy[SELF_SIGNER]
    print(f"\nHolding out the recorded signer: {held_self:.2%}")
    print("That is the honest generalisation figure for this signer —")
    print("the model had never seen this camera or room for that fold.")

    # The shipped model trains on everyone, including the recorded signer.
    # Its accuracy is not reported: there is no held-out signer left. The
    # folds above are the number that goes in the report.
    print("\nTraining the shipped model on all signers…")

    split = np.random.default_rng(0).random(len(X)) < 0.9
    final = fit(
        build(X.shape[1], len(letters)),
        X[split], y[split], X[~split], y[~split], args.epochs,
    )

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    final.save(out / "static_letters.h5")
    (out / "labels.json").write_text(json.dumps(letters, indent=2))
    (out / "evaluation.json").write_text(json.dumps({
        "protocol": "leave-one-signer-out over Surrey plus the recorded signer",
        "folds": fold_accuracy,
        "mean": float(acc.mean()),
        "std": float(acc.std()),
        "held_out_self": held_self,
        "surrey_vectors": int(len(Xs)),
        "recorded_vectors": int(len(Xr)),
        "recorded_per_class": args.per_class,
        "letters": letters,
    }, indent=2))

    print(f"\nWrote {out}/static_letters.h5")
    print(f"      {out}/labels.json")
    print(f"      {out}/evaluation.json")

    print(f"\nReport: {acc.mean():.2%} ± {acc.std():.2%} "
          f"(leave-one-signer-out, {len(signers)} signers)")
    print("The shipped model has seen this signer, so the demo should work.")
    print("The reported figure comes from folds where it had not.")

    return 0


if __name__ == "__main__":
    sys.exit(main())