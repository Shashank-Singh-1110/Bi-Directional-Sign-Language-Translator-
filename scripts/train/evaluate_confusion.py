"""
scripts/train/evaluate_confusion.py — Which letters get mistaken for which

Per-class accuracy says r scored 72.9%. It does not say what r was mistaken
for, and that is the part worth reporting: "r is read as u 18% of the time"
is a claim about the representation, while "r scored 72.9%" is a number.

The hypothesis this tests: r, u and v are all two-finger handshapes
separated by fine inter-finger detail, and wrist-origin normalisation
discards scale, which is some of what distinguishes them. If that is right,
the confusions should cluster inside that group rather than spread evenly.

Re-runs the same leave-one-signer-out folds as train_loso.py and pools every
prediction, so the matrix describes the honest protocol rather than a model
evaluated on signers it trained on.

    python scripts/train/evaluate_confusion.py data/processed/surrey.npz \
        --out Documents/evaluation
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402


def build(input_dim: int, classes: int):
    """Identical to train_loso.py — the matrix must describe that model."""
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("--out", default="Documents/evaluation")
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()

    data = np.load(pathlib.Path(args.npz), allow_pickle=True)
    X, y_raw, signer = data["X"], data["y"], data["signer"]

    labels = sorted(set(y_raw.tolist()))
    index = {label: i for i, label in enumerate(labels)}
    y = np.array([index[v] for v in y_raw], dtype=np.int64)
    signers = sorted(set(signer.tolist()))

    n = len(labels)
    matrix = np.zeros((n, n), dtype=np.int64)

    print(f"{len(X)} vectors · {n} classes · {len(signers)} signers")
    print("Pooling predictions across leave-one-signer-out folds.\n")

    for held in signers:
        test = signer == held
        model = fit(
            build(X.shape[1], n),
            X[~test], y[~test], X[test], y[test], args.epochs,
        )
        predicted = model.predict(X[test], verbose=0).argmax(axis=1)

        for truth, guess in zip(y[test], predicted):
            matrix[truth, guess] += 1

        correct = (predicted == y[test]).mean()
        print(f"  fold {held}: {correct:.2%}")

    total = matrix.sum()
    print(f"\nPooled accuracy {np.trace(matrix) / total:.2%} over {total} predictions\n")

    # The confusions, ranked. This is the output worth quoting.
    print("Most frequent confusions:")
    print(f"{'true':>6}{'predicted':>12}{'count':>9}{'of true':>10}")
    print("-" * 37)

    pairs = []
    for i in range(n):
        row_total = matrix[i].sum()
        if row_total == 0:
            continue
        for j in range(n):
            if i != j and matrix[i, j]:
                pairs.append((labels[i], labels[j], matrix[i, j],
                              matrix[i, j] / row_total))

    pairs.sort(key=lambda p: p[3], reverse=True)
    for true_label, guess, count, share in pairs[:15]:
        print(f"{true_label:>6}{guess:>12}{count:>9}{share:>9.1%}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # CSV for the report, with headers so it can be pasted into a table.
    csv = out / "surrey_confusion_matrix.csv"
    with csv.open("w") as fh:
        fh.write("true\\predicted," + ",".join(labels) + "\n")
        for i, label in enumerate(labels):
            fh.write(label + "," + ",".join(str(v) for v in matrix[i]) + "\n")

    (out / "surrey_confusions.json").write_text(json.dumps({
        "protocol": "leave-one-signer-out, predictions pooled",
        "pooled_accuracy": float(np.trace(matrix) / total),
        "labels": labels,
        "top_confusions": [
            {"true": t, "predicted": p, "count": int(c), "share": float(s)}
            for t, p, c, s in pairs[:30]
        ],
    }, indent=2))

    print(f"\nWrote {csv}")
    print(f"      {out}/surrey_confusions.json")

    # Does the two-finger hypothesis hold?
    group = {"r", "u", "v"}
    inside = sum(c for t, p, c, _ in pairs if t in group and p in group)
    leaving = sum(c for t, p, c, _ in pairs if t in group)

    if leaving:
        print(f"\nOf all errors on r, u and v, {inside / leaving:.1%} are")
        print("confusions with each other rather than with other letters.")
        print("These are the two-finger handshapes; a high share here means")
        print("the representation is losing the detail that separates them.")

    return 0


if __name__ == "__main__":
    sys.exit(main())