"""
scripts/convert/convert_models.py — Keras models to TFLite for deployment

The deployment dependency set excludes TensorFlow deliberately: the server
ships tflite-runtime at roughly 1 MB rather than TensorFlow at roughly
600 MB, and CI fails the build if TensorFlow, MediaPipe or OpenCV are
importable. Keras .h5 files cannot be loaded without TensorFlow, so every
model that reaches production has to be converted first.

Two models, two shapes:

  letters   (126,)      a dense classifier over one frame
  words     (30, 264)   two LSTM layers over a sequence

The LSTM is the awkward one. model.export() writes a serving signature with
a dynamic batch dimension regardless of the model's own fixed input shape;
the converter then cannot make the LSTM's TensorList element_shape static
and falls back to Select Ops, which would pull the large runtime back in.
Tracing the function with an explicit input_signature is what pins the batch
to 1. Keras 3 additionally holds weights as live resource variables, so they
are frozen to constants first.

Both conversions are verified against the original: predictions must agree
on every sample, or the conversion is rejected rather than shipped.

    python scripts/convert/convert_models.py
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402


def convert(model, input_shape, float16=True):
    """
    Trace, freeze and convert. Raises if builtin ops are not enough.

    input_shape excludes the batch dimension, which is pinned to 1.
    """
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, *input_shape], dtype=tf.float32)
    ])
    def serve(x):
        return model(x, training=False)

    concrete = serve.get_concrete_function()

    # Keras 3.15 leaves weights as resource variables; without this the
    # converter fails with "ReadVariableOp: missing attribute 'value'".
    frozen = convert_variables_to_constants_v2(concrete)
    frozen.graph.as_graph_def()

    conv = tf.lite.TFLiteConverter.from_concrete_functions([frozen], model)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    conv.experimental_enable_resource_variables = True

    if float16:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]

    return conv.convert()


def inspect(tflite_bytes):
    """Which operations the converted model actually uses."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()

    ops = sorted({d["op_name"] for d in interp._get_ops_details()})
    flex = [o for o in ops if o.lower().startswith("flex")]
    return ops, flex, interp


def agreement(model, tflite_bytes, samples):
    """
    Fraction of samples where the converted model picks the same class.

    Anything below 100% means the deployed model behaves differently from
    the one that was evaluated, which would make the reported accuracy a
    claim about a different artefact.
    """
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    original = model.predict(samples, verbose=0).argmax(axis=1)

    converted = []
    for row in samples:
        interp.set_tensor(inp["index"], row[None, ...].astype(np.float32))
        interp.invoke()
        converted.append(int(interp.get_tensor(out["index"])[0].argmax()))

    return float((original == np.array(converted)).mean())


def do_one(name, model_path, labels_path, out_path, input_shape,
           samples, float16=True):
    """Convert one model and report. Returns True on success."""
    from tensorflow import keras

    model_path = pathlib.Path(model_path)
    if not model_path.exists():
        print(f"{name}: no model at {model_path} — skipping\n")
        return False

    print(f"{name}")
    model = keras.models.load_model(model_path)

    try:
        blob = convert(model, input_shape, float16)
    except Exception as e:
        print(f"  conversion failed: {e}\n")
        return False

    ops, flex, _ = inspect(blob)

    if flex:
        print(f"  REJECTED — needs Select Ops: {flex}")
        print("  Shipping this would require the full TensorFlow runtime.\n")
        return False

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)

    size_kb = len(blob) // 1024
    print(f"  {out_path}  {size_kb} KB")
    print(f"  ops: {', '.join(ops)}")

    if samples is not None and len(samples):
        match = agreement(model, blob, samples)
        print(f"  agreement with the original: {match:.1%} "
              f"over {len(samples)} samples")
        if match < 1.0:
            print("  WARNING: below 100%. The deployed model does not behave")
            print("  identically to the one that was evaluated.")

    # The label order travels with the model. Getting it from a set
    # iteration once shifted every prediction by one class.
    labels_path = pathlib.Path(labels_path)
    if labels_path.exists():
        target = out_path.parent / "labels.json"
        if target.resolve() != labels_path.resolve():
            target.write_text(labels_path.read_text())
        labels = json.loads(labels_path.read_text())
        print(f"  {len(labels)} labels")

    print()
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--letters", default="models/static/static_letters.h5")
    ap.add_argument("--letters-labels", default="models/static/labels.json")
    ap.add_argument("--words", default="models/words/words.h5")
    ap.add_argument("--words-labels", default="models/words/labels.json")
    ap.add_argument("--out", default="models/tflite")
    ap.add_argument("--letters-data", default="data/processed/surrey.npz")
    ap.add_argument("--words-data", default="data/processed/wlasl100_full.npz")
    ap.add_argument("--no-float16", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    float16 = not args.no_float16

    rng = np.random.default_rng(0)
    ok = []

    # ── Letters: one frame, dense ──────────────────────────────────────
    letter_samples = None
    letters_data = pathlib.Path(args.letters_data)
    if letters_data.exists():
        X = np.load(letters_data, allow_pickle=True)["X"].astype(np.float32)
        idx = rng.choice(len(X), min(300, len(X)), replace=False)
        letter_samples = X[idx]

    ok.append(do_one(
        "Letters", args.letters, args.letters_labels,
        out / "letters_fp16.tflite", (126,), letter_samples, float16,
    ))

    # ── Words: 30 frames, LSTM ─────────────────────────────────────────
    word_samples = None
    words_data = pathlib.Path(args.words_data)
    if words_data.exists():
        X = np.load(words_data, allow_pickle=True)["X"].astype(np.float32)
        # The word model was trained with velocity appended.
        deltas = np.zeros_like(X)
        deltas[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
        X = np.concatenate([X, deltas], axis=2)
        idx = rng.choice(len(X), min(200, len(X)), replace=False)
        word_samples = X[idx]

    word_shape = (30, word_samples.shape[2]) if word_samples is not None else (30, 264)

    ok.append(do_one(
        "Words", args.words, args.words_labels,
        out / "words_fp16.tflite", word_shape, word_samples, float16,
    ))

    if all(ok):
        print("Both models converted with builtin operations only.")
        print("The deployment set can stay on tflite-runtime.")
        return 0

    print("At least one conversion did not succeed. Do not deploy until it")
    print("does, or the server will need the full TensorFlow runtime.")
    return 1


if __name__ == "__main__":
    sys.exit(main())