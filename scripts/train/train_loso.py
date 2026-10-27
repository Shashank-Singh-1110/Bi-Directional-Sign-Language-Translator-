"""
scripts/train/train_loso.py — Leave-one-signer-out training and evaluation

The recorded dataset's 98.96% came from a random split over near-duplicate
sequences, so train and test held copies of the same recordings. This holds
out whole people instead: train on four signers, test on the fifth, rotate
through all five.

That produces five independent accuracy figures rather than one, and their
spread says how much of the performance depends on who is signing. A single
random-split number cannot distinguish a model that learned handshapes from
one that learned a person.

The classifier is small on purpose. These are static letters, so each
example is one frame — no sequence model, no recurrence. The deployed LSTM
stays where it is, for the motion signs it was built for.

    python scripts/train/train_loso.py data/processed/surrey.npz \
        --out models/static

Writes, per fold, nothing permanent; at the end it retrains on all signers
and saves that model plus the label order. Label order is written to disk
because getting it from a set iteration once shifted every prediction by one
class and cost a day.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402


def build(input_dim: int, classes: int):
    """
    A small dense classifier over one frame of normalised landmarks.

    Two hidden layers is enough for 126 inputs and 24 classes; more capacity
    mostly buys a better fit to the training signers, which is the opposite
    of what is being measured here.
    """
    from tensorflow import keras

    return keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(classes, activation="softmax"),
    ])


def compile_and_fit(model, X_train, y_train, X_val, y_val, epochs, quiet):
    from tensorflow import keras

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=64,
        callbacks=[stop],
        verbose=0 if quiet else 1,
    )
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", help="Feature file from extract_landmarks.py")
    ap.add_argument("--out", default="models/static",
                    help="Where to write the final model and labels")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    path = pathlib.Path(args.npz)
    if not path.exists():
        print(f"No such file: {path}")
        return 1

    data = np.load(path, allow_pickle=True)
    X, y_raw, signer = data["X"], data["y"], data["signer"]

    # Sorted, and written to disk with the model. An implicit ordering from
    # a set once shifted every label by one class.
    labels = sorted(set(y_raw.tolist()))
    index = {label: i for i, label in enumerate(labels)}
    y = np.array([index[v] for v in y_raw], dtype=np.int64)

    signers = sorted(set(signer.tolist()))

    print(f"{len(X)} vectors · {len(labels)} classes · {len(signers)} signers")
    print(f"Classes: {' '.join(labels)}\n")

    if len(signers) < 2:
        print("Only one signer. A leave-one-signer-out evaluation is not")
        print("possible, and a random split here would measure template")
        print("recall — the problem this is meant to avoid.")
        return 1

    fold_accuracy = []
    per_class_hits = np.zeros(len(labels))
    per_class_total = np.zeros(len(labels))

    print(f"{'held out':<12}{'train':>9}{'test':>9}{'accuracy':>11}")
    print("-" * 41)

    for held in signers:
        test = signer == held
        train = ~test

        model = build(X.shape[1], len(labels))
        model = compile_and_fit(
            model, X[train], y[train], X[test], y[test],
            args.epochs, not args.verbose,
        )

        probs = model.predict(X[test], verbose=0)
        predicted = probs.argmax(axis=1)
        truth = y[test]

        accuracy = float((predicted == truth).mean())
        fold_accuracy.append(accuracy)

        for cls in range(len(labels)):
            mask = truth == cls
            per_class_total[cls] += mask.sum()
            per_class_hits[cls] += (predicted[mask] == cls).sum()

        print(f"{held:<12}{train.sum():>9}{test.sum():>9}{accuracy:>10.2%}")

    print("-" * 41)

    acc = np.array(fold_accuracy)
    print(f"{'mean':<12}{'':>9}{'':>9}{acc.mean():>10.2%}")
    print(f"{'std dev':<12}{'':>9}{'':>9}{acc.std():>10.2%}")
    print(f"{'worst':<12}{'':>9}{'':>9}{acc.min():>10.2%}")

    print("\nPer-class accuracy, pooled across folds:")
    order = np.argsort(per_class_hits / np.maximum(per_class_total, 1))
    for cls in order:
        total = per_class_total[cls]
        if total == 0:
            continue
        rate = per_class_hits[cls] / total
        bar = "#" * int(rate * 24)
        print(f"  {labels[cls]:<4}{rate:>7.1%}  {bar}")

    # Retrain on everyone for the deployable model. Its accuracy is not
    # reported, because there is no held-out signer left to measure it on —
    # the folds above are the honest number.
    print("\nRetraining on all signers for the shipped model…")

    split = np.random.default_rng(0).random(len(X)) < 0.9
    final = build(X.shape[1], len(labels))
    final = compile_and_fit(
        final, X[split], y[split], X[~split], y[~split],
        args.epochs, not args.verbose,
    )

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    final.save(out / "static_letters.h5")
    (out / "labels.json").write_text(json.dumps(labels, indent=2))
    (out / "evaluation.json").write_text(json.dumps({
        "folds": {s: a for s, a in zip(signers, fold_accuracy)},
        "mean": float(acc.mean()),
        "std": float(acc.std()),
        "worst": float(acc.min()),
        "per_class": {
            labels[c]: float(per_class_hits[c] / per_class_total[c])
            for c in range(len(labels)) if per_class_total[c]
        },
        "vectors": int(len(X)),
        "signers": signers,
        "protocol": "leave-one-signer-out",
    }, indent=2))

    print(f"\nWrote {out}/static_letters.h5")
    print(f"      {out}/labels.json")
    print(f"      {out}/evaluation.json")

    print(f"\nReport this as {acc.mean():.2%} ± {acc.std():.2%} "
          f"(leave-one-signer-out, {len(signers)} folds).")
    print("It is not comparable to the recorded dataset's 98.96%, which came")
    print("from a random split over near-duplicates. Different protocol,")
    print("different question.")

    return 0


if __name__ == "__main__":
    sys.exit(main())