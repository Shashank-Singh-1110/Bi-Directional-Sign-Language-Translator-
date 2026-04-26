import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import json

ACTIONS = np.array([
    'Hello', 'Thanks', 'Yes', 'I LOVE YOU', 'No', 'Sorry',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
])

DATA_PATH      = 'DATASET'
NORM_DATA_PATH = 'DATASET_NORM'
MODEL_SAVE     = 'action_norm.h5'
REPORT_DIR     = 'training_reports'
NO_SEQUENCES   = 30
SEQ_LENGTH     = 30
FEATURE_DIM    = 126


def normalize_hand(hand_63):
    pts = hand_63.reshape(21, 3)
    wrist = pts[0].copy()
    pts = pts - wrist

    dists = np.linalg.norm(pts, axis=1)
    max_d = dists.max()
    if max_d > 1e-6:
        pts = pts / max_d

    return pts.flatten()


def normalize_keypoints(raw_258):
    lh_raw = raw_258[132:195]
    rh_raw = raw_258[195:258]

    lh_norm = normalize_hand(lh_raw)
    rh_norm = normalize_hand(rh_raw)

    return np.concatenate([lh_norm, rh_norm])   # (126,)


def build_normalized_dataset():
    if os.path.exists(NORM_DATA_PATH):
        print(f"[NORM] {NORM_DATA_PATH} already exists — skipping rebuild.")
        print(f"       Delete {NORM_DATA_PATH} to force rebuild.")
        return

    print(f"[NORM] Building normalized dataset → {NORM_DATA_PATH}")
    os.makedirs(NORM_DATA_PATH, exist_ok=True)
    total = 0

    for action in ACTIONS:
        for seq in range(NO_SEQUENCES):
            out_dir = os.path.join(NORM_DATA_PATH, action, str(seq))
            os.makedirs(out_dir, exist_ok=True)
            for frame in range(SEQ_LENGTH):
                src = os.path.join(DATA_PATH, action, str(seq), f'{frame}.npy')
                dst = os.path.join(out_dir, f'{frame}.npy')
                if os.path.exists(src):
                    raw = np.load(src)
                    norm = normalize_keypoints(raw)
                    np.save(dst, norm)
                    total += 1

    print(f"[NORM] Done — {total} frames normalized to {FEATURE_DIM}-dim")


def load_normalized_data():
    print("\n[DATA] Loading normalized dataset...")
    label_map = {a: i for i, a in enumerate(ACTIONS)}
    sequences, labels = [], []

    for action in ACTIONS:
        count = 0
        for seq in range(NO_SEQUENCES):
            window = []
            for frame in range(SEQ_LENGTH):
                path = os.path.join(NORM_DATA_PATH, action, str(seq), f'{frame}.npy')
                if os.path.exists(path):
                    window.append(np.load(path))
            if len(window) == SEQ_LENGTH:
                sequences.append(window)
                labels.append(label_map[action])
                count += 1

    X = np.array(sequences)
    y = to_categorical(labels, num_classes=len(ACTIONS))
    print(f"[DATA] X: {X.shape}  y: {y.shape}")
    return X, y


def build_model():
    model = Sequential([
        Input(shape=(SEQ_LENGTH, FEATURE_DIM)),
        LSTM(128, return_sequences=True, activation='tanh'),
        LSTM(64,  return_sequences=False, activation='tanh'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(ACTIONS), activation='softmax'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    model.summary()
    return model

def train(model, X_train, y_train):
    y_int = np.argmax(y_train, axis=1)
    cw    = compute_class_weight('balanced',
                                  classes=np.unique(y_int), y=y_int)
    cw_dict = dict(enumerate(cw))

    os.makedirs(REPORT_DIR, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=50,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(REPORT_DIR, 'best_norm.h5'),
                        monitor='val_loss', save_best_only=True, verbose=0)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=500,
        validation_split=0.10,
        batch_size=16,
        callbacks=callbacks,
        class_weight=cw_dict,
        verbose=1
    )
    return history



if __name__ == '__main__':
    print("=" * 55)
    print("  NORMALIZE + RETRAIN")
    print(f"  Input dim : {FEATURE_DIM} (hand only, normalized)")
    print(f"  Classes   : {len(ACTIONS)}")
    print("=" * 55)

    build_normalized_dataset()

    X, y = load_normalized_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42,
        stratify=np.argmax(y, axis=1))
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    model   = build_model()
    history = train(model, X_train, y_train)
    model.save(MODEL_SAVE)
    print(f"\n[SAVED] {MODEL_SAVE}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc*100:.2f}%")
    print(classification_report(y_true, y_pred,
                                 target_names=ACTIONS, zero_division=0))

    with open(os.path.join(REPORT_DIR, 'metrics_norm.json'), 'w') as f:
        json.dump({'model': MODEL_SAVE,
                   'feature_dim': FEATURE_DIM,
                   'normalized': True,
                   'test_accuracy': float(acc)}, f, indent=2)