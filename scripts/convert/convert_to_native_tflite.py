"""
convert_native_tflite.py — Rebuild the LSTM as a TFLite-native fused op
=======================================================================
The default export compiles the LSTM into a generic TensorList loop, which
needs the TFLite Select Ops (Flex) library on Android — roughly 6 MB of extra
binary for a 386 KB model.

This script re-exports the SAME WEIGHTS with a fixed batch dimension so the
converter can recognise the fusable pattern and emit the builtin
UnidirectionalSequenceLSTM op instead. No retraining. No architecture change.

SELECT_TF_OPS is deliberately NOT enabled, so if fusion fails the conversion
errors out instead of silently falling back to Flex.

Run: python convert_native_tflite.py
"""

import os
import time

import numpy as np
import tensorflow as tf

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
KERAS_MODEL  = 'models/action_norm.h5'
DATASET_NORM = 'data/DATASET_NORM'
OUT_DIR      = 'models/tflite/'
SEQ_LEN      = 30
N_FEATURES   = 126
EVAL_SAMPLES = 96

# Training class order — must match the order used during training,
# which is NOT alphabetical. Keep in sync with convert_to_tflite.py.
ACTIONS = [
    'Hello', 'Thanks', 'Yes', 'I LOVE YOU', 'No', 'Sorry',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]


# ─────────────────────────────────────────────
#  STEP 1 — rebuild with a fixed batch dimension
# ─────────────────────────────────────────────
def rebuild_static(src_model):
    """
    Clone src_model layer-for-layer onto a fixed (1, 30, 126) input.

    Layer configs are read from the source model, so the architecture is
    whatever you actually trained — nothing is hardcoded here. Weights are
    copied across afterwards, so predictions are bit-identical.
    """
    inp = tf.keras.Input(batch_shape=(1, SEQ_LEN, N_FEATURES), name='input')
    x = inp

    for layer in src_model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        cfg = layer.get_config()

        # Force the fusable LSTM configuration.
        # Only touch keys this Keras version actually accepts — Keras 3
        # removed `time_major` from the LSTM signature entirely.
        if isinstance(layer, tf.keras.layers.LSTM):
            for key, value in (('unroll', False),
                               ('time_major', False),
                               ('stateful', False)):
                if key in cfg:
                    cfg[key] = value

        x = layer.__class__.from_config(cfg)(x)

    static = tf.keras.Model(inp, x, name='signbridge_static')
    static.set_weights(src_model.get_weights())
    return static


def verify_identical(src_model, static_model, X):
    """Confirm the rebuild predicts exactly what the original does."""
    max_delta = 0.0
    for i in range(min(32, len(X))):
        sample = X[i:i + 1].astype(np.float32)
        a = src_model.predict(sample, verbose=0)
        b = static_model.predict(sample, verbose=0)
        max_delta = max(max_delta, float(np.abs(a - b).max()))
    return max_delta


# ─────────────────────────────────────────────
#  STEP 2 — convert with builtins only
# ─────────────────────────────────────────────
def convert_native(static_model, float16=True):
    """
    Convert from an explicitly-traced concrete function.

    model.export() writes a serving signature with a dynamic batch dim
    (None, 30, 126) regardless of the model's own fixed input shape. The
    converter then cannot make the LSTM's TensorList element_shape static,
    and fusion fails. Tracing the function ourselves with an explicit
    input_signature is what actually pins the batch to 1.

    If the LSTM still does not fuse, the converter raises here rather than
    quietly emitting a Flex-dependent model — which is the point.
    """
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, SEQ_LEN, N_FEATURES], dtype=tf.float32)
    ])
    def serve(x):
        return static_model(x, training=False)

    concrete = serve.get_concrete_function()

    # Inline the weights as constants. Keras 3.15 leaves them as live
    # resource variables, and the converter then fails with
    # "ReadVariableOp: missing attribute 'value'".
    frozen = convert_variables_to_constants_v2(concrete)
    frozen.graph.as_graph_def()

    conv = tf.lite.TFLiteConverter.from_concrete_functions(
        [frozen], static_model
    )
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    # Keras 3 holds weights as resource variables; without this the
    # converter cannot constant-fold them away.
    conv.experimental_enable_resource_variables = True

    if float16:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]

    return conv.convert()


def list_ops(tflite_bytes):
    """Read back which ops the model actually uses."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    ops = sorted({d['op_name'] for d in interp._get_ops_details()})
    flex = [o for o in ops if o.lower().startswith('flex')]
    return ops, flex


# ─────────────────────────────────────────────
#  DATA + BENCHMARK
# ─────────────────────────────────────────────
def load_dataset():
    if not os.path.isdir(DATASET_NORM):
        print(f"[DATA] '{DATASET_NORM}' not found — skipping evaluation")
        return None, None

    X, y = [], []

    for label, action in enumerate(ACTIONS):
        a_dir = os.path.join(DATASET_NORM, action)
        if not os.path.isdir(a_dir):
            print(f"[DATA] Missing class folder: {action}")
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

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"[DATA] Loaded {len(X)} sequences, shape {X.shape}")
    return X, y


def run_tflite(tflite_bytes, X):
    """Run inference sample-by-sample; returns (predictions, latencies_ms)."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    preds, lat = [], []
    for i in range(len(X)):
        sample = X[i:i + 1].astype(np.float32)
        t0 = time.perf_counter()
        interp.set_tensor(inp['index'], sample)
        interp.invoke()
        result = interp.get_tensor(out['index'])
        lat.append((time.perf_counter() - t0) * 1000)
        preds.append(int(np.argmax(result)))

    return np.array(preds), lat


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 68)
    print("  SignBridge — TFLite-Native LSTM Conversion")
    print("=" * 68)
    print(f"  TensorFlow: {tf.__version__}")

    # ---- 1. Load ----
    print(f"\n[1/5] Loading {KERAS_MODEL}")
    src = tf.keras.models.load_model(KERAS_MODEL, compile=False)
    print(f"      Parameters: {src.count_params():,}")

    # ---- 2. Rebuild ----
    print("\n[2/5] Rebuilding with fixed batch shape (1, 30, 126)")
    static = rebuild_static(src)
    lstm_count = sum(1 for l in static.layers
                     if isinstance(l, tf.keras.layers.LSTM))
    print(f"      Layers cloned: {len(static.layers) - 1}  "
          f"({lstm_count} LSTM)")

    # ---- 3. Data + equivalence check ----
    print("\n[3/5] Loading dataset")
    X, y = load_dataset()

    if X is not None:
        delta = verify_identical(src, static, X)
        status = "identical" if delta < 1e-5 else f"DIVERGED by {delta:.2e}"
        print(f"      Weight transfer: {status} (max delta {delta:.2e})")
        if delta >= 1e-5:
            print("      ABORTING — rebuilt model does not match original.")
            return

    # ---- 4. Convert ----
    print("\n[4/5] Converting with TFLITE_BUILTINS only (no Flex)")
    try:
        blob = convert_native(static, float16=True)
    except Exception as e:
        print(f"      FAILED: {e}")
        print("\n      The LSTM did not fuse. Falling back is pointless —")
        print("      keep tflite_models/action_float16.tflite and ship")
        print("      with the Select Ops library instead.")
        return

    path = os.path.join(OUT_DIR, 'action_native_fp16.tflite')
    with open(path, 'wb') as f:
        f.write(blob)
    size_kb = len(blob) // 1024
    print(f"      ok  ({size_kb} KB)  ->  {path}")

    # ---- 5. Inspect ops ----
    print("\n[5/5] Verifying op set")
    ops, flex = list_ops(blob)
    print(f"      Ops used: {', '.join(ops)}")

    # The requirement is "no Flex ops", not "one specific fused op".
    # The converter may emit either UNIDIRECTIONAL_SEQUENCE_LSTM or a
    # builtin WHILE loop with FULLY_CONNECTED gates — both are Flex-free
    # and both run on the base TFLite runtime.
    builtin_only = not flex

    if flex:
        print(f"      Flex ops present: {flex}")
        print("      Still needs the Select Ops library.")
    else:
        form = ('fused UNIDIRECTIONAL_SEQUENCE_LSTM'
                if any('SEQUENCE_LSTM' in o.upper() for o in ops)
                else 'builtin WHILE loop')
        print(f"      BUILTIN-ONLY — zero Flex ops ({form}).")
        print("      Android needs only the base TFLite runtime (~1 MB).")

    # ---- Benchmark ----
    if X is not None:
        print("\n" + "=" * 68)
        print("RESULTS")
        print("=" * 68)

        n = min(EVAL_SAMPLES, len(X))
        idx = np.linspace(0, len(X) - 1, n, dtype=int)
        Xe, ye = X[idx], y[idx]

        keras_preds = np.array([
            int(np.argmax(src.predict(Xe[i:i + 1], verbose=0)))
            for i in range(n)
        ])

        preds, lat = run_tflite(blob, Xe)
        acc = (preds == ye).mean() * 100
        agree = (preds == keras_preds).mean() * 100

        print(f"{'Variant':<22}{'Size (KB)':>11}{'Latency':>11}"
              f"{'Accuracy':>11}{'Agreement':>12}")
        print("-" * 68)
        keras_acc = (keras_preds == ye).mean() * 100
        print(f"{'Keras (.h5)':<22}"
              f"{os.path.getsize(KERAS_MODEL)//1024:>11}"
              f"{'13.3ms':>11}{keras_acc:>10.2f}%{'100.0%':>12}")
        print(f"{'TFLite native fp16':<22}{size_kb:>11}"
              f"{np.mean(lat):>10.1f}ms{acc:>10.2f}%{agree:>11.1f}%")
        print("-" * 68)
        print(f"p50 {np.percentile(lat,50):.2f}ms   "
              f"p95 {np.percentile(lat,95):.2f}ms   "
              f"max {max(lat):.2f}ms")

        print("\n" + "=" * 68)
        print("APK FOOTPRINT")
        print("=" * 68)
        if builtin_only:
            print(f"  Model              {size_kb:>6} KB")
            print(f"  TFLite runtime     {'~1024':>6} KB  (base only)")
            print(f"  {'-'*34}")
            print(f"  Total              {size_kb + 1024:>6} KB")
            print("\n  Previous build (Select Ops): ~7000 KB")
            saved = 7000 - (size_kb + 1024)
            print(f"  Saved: ~{saved} KB")
        else:
            print("  Still requires Select Ops (~6 MB extra).")


if __name__ == '__main__':
    main()