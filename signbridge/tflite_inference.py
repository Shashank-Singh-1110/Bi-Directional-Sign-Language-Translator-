import os
import time
import numpy as np

try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    # tflite_runtime is a much smaller install for deployment
    from tflite_runtime.interpreter import Interpreter

# Builtin-ops-only model — no Select Ops / Flex delegate required.
DEFAULT_MODEL = 'tflite_models/action_native_fp16.tflite'


class TFLiteModel:
    """Wraps a TFLite interpreter with a Keras-compatible predict() method."""

    def __init__(self, model_path: str = DEFAULT_MODEL, num_threads: int = 2):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} not found — run convert_native_tflite.py first"
            )
        self.model_path = model_path
        self.interpreter = Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()

        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_index    = self.input_details[0]['index']
        self.output_index   = self.output_details[0]['index']
        self.input_dtype    = self.input_details[0]['dtype']
        self.input_shape    = self.input_details[0]['shape']

        # Running latency stats
        self._times = []

        print(f"[TFLITE] Loaded {model_path}")
        print(f"[TFLITE] Input  {self.input_shape} {np.dtype(self.input_dtype).name}")
        print(f"[TFLITE] Output {self.output_details[0]['shape']}")

    def predict(self, X, verbose=0):
        """
        Keras-compatible predict.

        Args:
            X: numpy array of shape (batch, 30, 126). Batch is processed
               one sample at a time since TFLite has a fixed batch size of 1.
        Returns:
            numpy array of shape (batch, 32) — softmax probabilities
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[np.newaxis, ...]

        results = []
        for i in range(X.shape[0]):
            sample = X[i:i+1].astype(self.input_dtype)
            self.interpreter.set_tensor(self.input_index, sample)

            t0 = time.perf_counter()
            self.interpreter.invoke()
            self._times.append((time.perf_counter() - t0) * 1000)

            results.append(self.interpreter.get_tensor(self.output_index)[0])

        return np.array(results)

    def __call__(self, X, **kwargs):
        return self.predict(X)

    @property
    def latency_ms(self) -> float:
        """Average inference latency over all calls so far."""
        return float(np.mean(self._times)) if self._times else 0.0

    @property
    def last_latency_ms(self) -> float:
        return self._times[-1] if self._times else 0.0

    def reset_stats(self):
        self._times = []

    def stats(self) -> dict:
        if not self._times:
            return {"calls": 0}
        t = np.array(self._times)
        return {
            "calls":  len(t),
            "mean_ms": round(float(t.mean()), 2),
            "p50_ms":  round(float(np.percentile(t, 50)), 2),
            "p95_ms":  round(float(np.percentile(t, 95)), 2),
            "max_ms":  round(float(t.max()), 2),
        }


# ── Verification utility ───────────────────────────────────────────────
def verify_against_keras(tflite_path: str, keras_path: str,
                         n_samples: int = 50, data_dir: str = "DATASET_NORM"):
    """
    Confirm the TFLite model produces the same predictions as Keras.
    Run this after conversion before deploying.
    """
    import tensorflow as tf

    ACTIONS = np.array([
        'Hello', 'Thanks', 'Yes', 'I LOVE YOU', 'No', 'Sorry',
        'A','B','C','D','E','F','G','H','I','J',
        'K','L','M','N','O','P','Q','R','S','T',
        'U','V','W','X','Y','Z'
    ])

    print("=" * 60)
    print("TFLite vs Keras Verification")
    print("=" * 60)

    keras_model  = tf.keras.models.load_model(keras_path)
    tflite_model = TFLiteModel(tflite_path)

    # Load some real sequences
    X, y = [], []
    for label_idx, action in enumerate(ACTIONS):
        d = os.path.join(data_dir, action)
        if not os.path.isdir(d):
            continue
        for seq in sorted(os.listdir(d))[:2]:      # 2 per class
            sd = os.path.join(d, seq)
            if not os.path.isdir(sd):
                continue
            window = []
            for f in range(30):
                fp = os.path.join(sd, f"{f}.npy")
                if not os.path.exists(fp):
                    break
                window.append(np.load(fp))
            if len(window) == 30:
                X.append(window)
                y.append(label_idx)

    if not X:
        print("No data found — using random input")
        X = np.random.randn(n_samples, 30, 126).astype(np.float32)
        y = None
    else:
        X = np.array(X, dtype=np.float32)[:n_samples]
        y = np.array(y)[:n_samples]

    print(f"\nTesting on {len(X)} sequences\n")

    keras_preds  = np.argmax(keras_model.predict(X, verbose=0), axis=1)
    tflite_preds = np.argmax(tflite_model.predict(X), axis=1)

    agreement = float(np.mean(keras_preds == tflite_preds)) * 100
    print(f"Agreement with Keras: {agreement:.2f}%")

    if y is not None:
        keras_acc  = float(np.mean(keras_preds == y)) * 100
        tflite_acc = float(np.mean(tflite_preds == y)) * 100
        print(f"Keras accuracy:       {keras_acc:.2f}%")
        print(f"TFLite accuracy:      {tflite_acc:.2f}%")
        print(f"Accuracy delta:       {tflite_acc - keras_acc:+.2f}%")

    print(f"\nTFLite latency: {tflite_model.stats()}")

    # Show any disagreements
    mismatches = np.where(keras_preds != tflite_preds)[0]
    if len(mismatches):
        print(f"\n{len(mismatches)} disagreements:")
        for i in mismatches[:10]:
            print(f"  sample {i}: Keras={ACTIONS[keras_preds[i]]:<12} "
                  f"TFLite={ACTIONS[tflite_preds[i]]}")
    else:
        print("\nNo disagreements — TFLite output matches Keras exactly")

    print("\n" + "=" * 60)
    if agreement >= 99:
        print("PASS — safe to deploy")
    elif agreement >= 95:
        print("MARGINAL — consider float16 instead of int8")
    else:
        print("FAIL — quantization is degrading the model too much")

    return agreement


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        verify_against_keras(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        # Just load and report info
        m = TFLiteModel(sys.argv[1])
        dummy = np.random.randn(10, 30, 126).astype(np.float32)
        m.predict(dummy)
        print(f"\nLatency over 10 runs: {m.stats()}")
    else:
        print("Usage:")
        print("  python tflite_inference.py <tflite_path>                  # info + latency")
        print("  python tflite_inference.py <tflite_path> <keras_path>     # verify")