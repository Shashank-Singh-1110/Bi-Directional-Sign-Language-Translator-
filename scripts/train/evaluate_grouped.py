import json
import os

import numpy as np

MODEL_PATH = 'models/action_norm.h5'
DATA_PATH  = 'data/DATASET_NORM'
OUT_DIR    = 'documents/evaluation'
SEQ_LEN    = 30
HOLDOUT_FROM = 24

ACTIONS = [
    'Hello', 'Thanks', 'Yes', 'I LOVE YOU', 'No', 'Sorry',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]


def load_with_groups():
    X, y, seq_ids = [], [], []

    for label, action in enumerate(ACTIONS):
        a_dir = os.path.join(DATA_PATH, action)
        if not os.path.isdir(a_dir):
            print(f"[WARN] missing class folder: {action}")
            continue

        for seq in sorted(os.listdir(a_dir), key=lambda s: int(s) if s.isdigit() else -1):
            s_dir = os.path.join(a_dir, seq)
            if not os.path.isdir(s_dir) or not seq.isdigit():
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
                seq_ids.append(int(seq))

    return (np.array(X, dtype=np.float32),
            np.array(y),
            np.array(seq_ids))


def score(model, X, y_true, label):
    """Predict and summarize."""
    probs = model.predict(X, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    conf = probs.max(axis=1) * 100
    correct = y_pred == y_true

    print(f"\n  {label}")
    print(f"    Sequences      {len(y_true)}")
    print(f"    Accuracy       {correct.mean() * 100:.2f}%  "
          f"({correct.sum()}/{len(y_true)})")
    print(f"    Confidence     {conf[correct].mean():.1f}% mean when correct")
    if (~correct).any():
        print(f"                   {conf[~correct].mean():.1f}% mean when wrong")
    print(f"    Lowest correct {conf[correct].min():.1f}%")

    return {
        'sequences': int(len(y_true)),
        'accuracy': round(float(correct.mean() * 100), 2),
        'correct': int(correct.sum()),
        'mean_confidence_correct': round(float(conf[correct].mean()), 2),
        'mean_confidence_wrong': (
            round(float(conf[~correct].mean()), 2) if (~correct).any() else None
        ),
        'min_confidence_correct': round(float(conf[correct].min()), 2),
        'y_pred': y_pred,
        'y_true': y_true,
        'probs': probs,
    }


def main():
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 66)
    print("  Evaluation — random split vs grouped split")
    print("=" * 66)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    X, y, seq_ids = load_with_groups()
    print(f"\n[DATA] {len(X)} sequences, {len(set(y))} classes")
    print(f"[DATA] Recording indices {seq_ids.min()}–{seq_ids.max()}")

    # ── Split A: random, as used in training ────────────────────────────
    _, X_rand, _, y_rand = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    rand = score(model, X_rand, y_rand, "RANDOM SPLIT (leaky)")

    # ── Split B: grouped by recording order ─────────────────────────────
    mask = seq_ids >= HOLDOUT_FROM
    X_grp, y_grp = X[mask], y[mask]

    if len(X_grp) == 0:
        print(f"\n[ERROR] No sequences numbered >= {HOLDOUT_FROM}.")
        return

    grouped = score(model, X_grp, y_grp,
                    f"GROUPED SPLIT (sequences {HOLDOUT_FROM}+, held out)")

    gap = rand['accuracy'] - grouped['accuracy']

    # ── Per-class report on the grouped split ───────────────────────────
    present = sorted(set(grouped['y_true']) | set(grouped['y_pred']))
    names = [ACTIONS[i] for i in present]

    report_txt = classification_report(
        grouped['y_true'], grouped['y_pred'],
        labels=present, target_names=names, zero_division=0
    )

    with open(os.path.join(OUT_DIR, 'grouped_split_report.txt'), 'w') as fh:
        fh.write("SignBridge — grouped-split evaluation\n")
        fh.write(f"Held out: sequences {HOLDOUT_FROM}+ of each class "
                 f"({grouped['sequences']} sequences)\n\n")
        fh.write(f"Random split accuracy  : {rand['accuracy']:.2f}%\n")
        fh.write(f"Grouped split accuracy : {grouped['accuracy']:.2f}%\n")
        fh.write(f"Gap                    : {gap:.2f} points\n\n")
        fh.write("The gap measures how much the random split was inflated by\n")
        fh.write("near-duplicate sequences appearing in both train and test.\n\n")
        fh.write(report_txt)

    cm = confusion_matrix(grouped['y_true'], grouped['y_pred'], labels=present)
    np.savetxt(os.path.join(OUT_DIR, 'grouped_confusion_matrix.csv'),
               cm, fmt='%d', delimiter=',',
               header=','.join(names), comments='')

    # ── Which signs actually fail ───────────────────────────────────────
    errors = {}
    for t, p in zip(grouped['y_true'], grouped['y_pred']):
        if t != p:
            key = f"{ACTIONS[t]} -> {ACTIONS[p]}"
            errors[key] = errors.get(key, 0) + 1

    with open(os.path.join(OUT_DIR, 'grouped_metrics.json'), 'w') as fh:
        json.dump({
            'model': MODEL_PATH,
            'holdout_from_sequence': HOLDOUT_FROM,
            'random_split': {
                k: v for k, v in rand.items()
                if k not in ('y_pred', 'y_true', 'probs')
            },
            'grouped_split': {
                k: v for k, v in grouped.items()
                if k not in ('y_pred', 'y_true', 'probs')
            },
            'inflation_points': round(float(gap), 2),
            'confusions': errors,
        }, fh, indent=2)

    # ── Verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("VERDICT")
    print("=" * 66)
    print(f"  Random split   {rand['accuracy']:.2f}%")
    print(f"  Grouped split  {grouped['accuracy']:.2f}%")
    print(f"  Inflation      {gap:.2f} points")

    if gap < 1:
        print("\n  Negligible gap. The random split was not meaningfully")
        print("  leaking — the model generalizes across recording order.")
    elif gap < 5:
        print("\n  Small gap. Some leakage, but the model holds up on")
        print("  sequences recorded later. Report the grouped number.")
    else:
        print("\n  Substantial gap. The random split was measuring memory,")
        print("  not generalization. The grouped number is the honest one,")
        print("  and it is the one to put in the paper.")

    if errors:
        print(f"\n  Confusions on the grouped split:")
        for pair, n in sorted(errors.items(), key=lambda kv: -kv[1]):
            print(f"    {pair:<28} {n}x")

    print(f"\n  Written to {OUT_DIR}/")

    print("\n  Caveat worth stating in the report: even the grouped split")
    print("  uses one signer in one session. It measures robustness to")
    print("  drift within a session, not across people. A second signer")
    print("  is the only way to measure that.")


if __name__ == '__main__':
    main()