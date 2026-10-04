import json
import os

import numpy as np

MODEL_PATH  = 'models/action_norm.h5'
DATA_PATH   = 'data/DATASET_NORM'
OUT_DIR     = 'documents/evaluation'
SEQ_LEN     = 30

TEST_SIZE    = 0.10
RANDOM_STATE = 42

ACTIONS = [
    'Hello', 'Thanks', 'Yes', 'I LOVE YOU', 'No', 'Sorry',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]


def load_data():
    X, y = [], []
    for label, action in enumerate(ACTIONS):
        a_dir = os.path.join(DATA_PATH, action)
        if not os.path.isdir(a_dir):
            print(f"[WARN] missing class folder: {action}")
            continue

        for seq in sorted(os.listdir(a_dir)):
            s_dir = os.path.join(a_dir, seq)
            if not os.path.isdir(s_dir):
                continue

            frames = []
            for f in range(SEQ_LEN):
                fp = os.path.join(s_dir, f'{f}.npy')
                if not os.path.exists(fp):
                    break
                frames.append(np.load(fp))

            if len(frames) == SEQ_LEN:
                X.append(frames)
                y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)


def main():
    import tensorflow as tf
    from sklearn.metrics import (classification_report, confusion_matrix)
    from sklearn.model_selection import train_test_split

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 64)
    print("  Evaluation — regenerating training_reports/ artifacts")
    print("=" * 64)

    print(f"\n[1/4] Loading {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"      {model.count_params():,} parameters")

    print(f"\n[2/4] Loading {DATA_PATH}")
    X, y = load_data()
    print(f"      {len(X)} sequences, shape {X.shape}")

    # Same split as training, so the test set is identical
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=len(ACTIONS))
    _, X_test, _, y_test = train_test_split(
        X, y_onehot,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    y_true = np.argmax(y_test, axis=1)
    print(f"      Held-out test set: {len(X_test)} sequences")

    print("\n[3/4] Predicting")
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    accuracy = (y_pred == y_true).mean() * 100
    print(f"      Accuracy: {accuracy:.2f}%  "
          f"({(y_pred == y_true).sum()}/{len(y_true)})")

    print("\n[4/4] Writing artifacts")

    # ---- Per-class report ----
    present = sorted(set(y_true) | set(y_pred))
    names = [ACTIONS[i] for i in present]

    report_txt = classification_report(
        y_true, y_pred, labels=present, target_names=names, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, labels=present, target_names=names,
        zero_division=0, output_dict=True
    )

    with open(os.path.join(OUT_DIR, 'classification_report.txt'), 'w') as fh:
        fh.write(f"SignBridge — held-out test set ({len(y_true)} sequences)\n")
        fh.write(f"Accuracy: {accuracy:.2f}%\n\n")
        fh.write(report_txt)

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_true, y_pred, labels=present)
    np.savetxt(os.path.join(OUT_DIR, 'confusion_matrix.csv'),
               cm, fmt='%d', delimiter=',',
               header=','.join(names), comments='')

    # ---- Misclassifications, named ----
    errors = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t != p:
            errors.append({
                'index': int(i),
                'true': ACTIONS[t],
                'predicted': ACTIONS[p],
                'confidence': round(float(probs[i][p]) * 100, 2),
                'true_class_confidence': round(float(probs[i][t]) * 100, 2),
            })

    # ---- Confidence distribution ----
    top_conf = probs.max(axis=1) * 100
    correct_mask = y_pred == y_true

    metrics = {
        'model': MODEL_PATH,
        'parameters': int(model.count_params()),
        'test_sequences': int(len(y_true)),
        'test_accuracy': round(float(accuracy), 2),
        'split': {
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE,
            'stratified': True,
        },
        'confidence': {
            'mean_correct': round(float(top_conf[correct_mask].mean()), 2),
            'mean_incorrect': (
                round(float(top_conf[~correct_mask].mean()), 2)
                if (~correct_mask).any() else None
            ),
            'min_correct': round(float(top_conf[correct_mask].min()), 2),
        },
        'misclassifications': errors,
        'per_class': {
            k: v for k, v in report_dict.items()
            if isinstance(v, dict)
        },
    }

    with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)

    # ---- Console summary ----
    print(f"      -> {OUT_DIR}/classification_report.txt")
    print(f"      -> {OUT_DIR}/confusion_matrix.csv")
    print(f"      -> {OUT_DIR}/metrics.json")

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"Accuracy          {accuracy:.2f}%")
    print(f"Mean confidence   {top_conf[correct_mask].mean():.1f}% when correct")
    if (~correct_mask).any():
        print(f"                  {top_conf[~correct_mask].mean():.1f}% when wrong")
    print(f"Lowest correct    {top_conf[correct_mask].min():.1f}%")

    if errors:
        print(f"\n{len(errors)} misclassification(s):")
        for e in errors:
            print(f"  {e['true']:<12} -> {e['predicted']:<12} "
                  f"at {e['confidence']:.1f}% "
                  f"(true class scored {e['true_class_confidence']:.1f}%)")
    else:
        print("\nNo misclassifications.")

    print("\nNote: the confidence gap between correct and incorrect")
    print("predictions is what the 25-point gate in the live pipeline")
    print("exploits — it rejects exactly this kind of uncertain output.")


if __name__ == '__main__':
    main()