package com.signbridge.ml

import android.content.Context
import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Sign classification with the same gating as the desktop pipeline.
 *
 * Feed it one 126-dim frame at a time via [addFrame]. It keeps a 30-frame
 * ring buffer, runs the model once the buffer is full, and only emits a
 * detection when four gates agree. Gating is the reason the live system
 * behaves: raw per-frame confidence is not trustworthy on its own.
 *
 * No camera dependency beyond loading the model from assets, so the whole
 * gating path can be driven from unit tests with synthetic frames.
 */
class SignClassifier(context: Context) {

    companion object {
        private const val MODEL_ASSET = "action_native_fp16.tflite"

        const val SEQUENCE_LENGTH = 30
        const val FEATURE_DIM = 126
        const val NUM_CLASSES = 32

        /** Frames examined by the motion gate. */
        private const val MOTION_WINDOW = 6

        /** Below this standard deviation the hand counts as settled. */
        private const val MOTION_THRESHOLD = 0.012f

        /** Word signs are more distinctive, so they clear at a lower bar. */
        private const val CONF_THRESHOLD_WORD = 75f
        private const val CONF_THRESHOLD_LETTER = 92f

        /** Top prediction must beat the runner-up by this many points. */
        private const val CONFIDENCE_GAP = 25f

        /** Consecutive agreeing predictions required before emitting. */
        private const val STABILITY_FRAMES = 10

        /** Silence after a detection, so a held sign fires once. */
        private const val COOLDOWN_MS = 3000L

        /** Training class order — NOT alphabetical. */
        val ACTIONS = arrayOf(
            "Hello", "Thanks", "Yes", "I LOVE YOU", "No", "Sorry",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
            "U", "V", "W", "X", "Y", "Z",
        )

        /** The first six are words; the rest are fingerspelled letters. */
        private const val FIRST_LETTER_INDEX = 6
    }

    /** Why a frame did or did not produce a detection. */
    sealed class Result {
        /** Buffer still filling. */
        data class Warming(val framesHeld: Int) : Result()

        /** A gate rejected it. [reason] is for the debug overlay. */
        data class Rejected(
            val reason: String,
            val topSign: String,
            val confidence: Float,
        ) : Result()

        /** All gates passed and the sign was stable. */
        data class Detected(
            val sign: String,
            val confidence: Float,
            val inferenceMs: Long,
        ) : Result()
    }

    private val interpreter: Interpreter

    private val buffer = ArrayDeque<FloatArray>(SEQUENCE_LENGTH)
    private val input = Array(1) { Array(SEQUENCE_LENGTH) { FloatArray(FEATURE_DIM) } }
    private val output = Array(1) { FloatArray(NUM_CLASSES) }

    private var stableSign = -1
    private var stableCount = 0
    private var lastEmitAt = 0L

    init {
        interpreter = Interpreter(loadModel(context), Interpreter.Options().apply {
            numThreads = 2
        })
    }

    private fun loadModel(context: Context): MappedByteBuffer {
        val fd = context.assets.openFd(MODEL_ASSET)
        FileInputStream(fd.fileDescriptor).use { stream ->
            return stream.channel.map(
                FileChannel.MapMode.READ_ONLY,
                fd.startOffset,
                fd.declaredLength,
            )
        }
    }

    /**
     * Add one frame and get the verdict.
     *
     * @param features   126 normalized values from [HandLandmarkerHelper]
     * @param handsFound 0 clears the buffer — a sign cannot span a gap
     *                   where the hands left the frame
     */
    fun addFrame(features: FloatArray, handsFound: Int): Result {
        require(features.size == FEATURE_DIM) {
            "Expected $FEATURE_DIM features, got ${features.size}"
        }

        // ── Gate 1: hand presence ────────────────────────────────────────
        if (handsFound == 0) {
            reset()
            return Result.Warming(0)
        }

        buffer.addLast(features)
        if (buffer.size > SEQUENCE_LENGTH) buffer.removeFirst()

        if (buffer.size < SEQUENCE_LENGTH) {
            return Result.Warming(buffer.size)
        }

        // ── Gate 2: motion ───────────────────────────────────────────────
        // A sign is read when the hand has settled into its shape. Mid
        // transition the model sees a blend of two signs and will happily
        // report something confident and wrong.
        if (!isStill()) {
            return Result.Rejected("moving", "", 0f)
        }

        // ── Inference ────────────────────────────────────────────────────
        val start = SystemClock.elapsedRealtimeNanos()

        buffer.forEachIndexed { i, frame -> frame.copyInto(input[0][i]) }
        interpreter.run(input, output)

        val elapsedMs = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000

        val probs = output[0]
        var top = 0
        var second = 0
        for (i in probs.indices) {
            if (probs[i] > probs[top]) {
                second = top
                top = i
            } else if (i != top && probs[i] > probs[second]) {
                second = i
            }
        }

        val topConf = probs[top] * 100f
        val secondConf = probs[second] * 100f
        val sign = ACTIONS[top]

        // ── Gate 3: confidence ───────────────────────────────────────────
        val threshold = if (top >= FIRST_LETTER_INDEX) {
            CONF_THRESHOLD_LETTER
        } else {
            CONF_THRESHOLD_WORD
        }

        if (topConf < threshold) {
            stableCount = 0
            return Result.Rejected("low confidence", sign, topConf)
        }

        // ── Gate 4: confidence gap ───────────────────────────────────────
        // Separates the visually similar pairs — V/Z, O/E, V/W — where the
        // model is confident about two classes at once.
        if (topConf - secondConf < CONFIDENCE_GAP) {
            stableCount = 0
            return Result.Rejected(
                "ambiguous (${ACTIONS[second]})", sign, topConf
            )
        }

        // ── Stability ────────────────────────────────────────────────────
        if (top == stableSign) {
            stableCount++
        } else {
            stableSign = top
            stableCount = 1
        }

        if (stableCount < STABILITY_FRAMES) {
            return Result.Rejected(
                "stabilising $stableCount/$STABILITY_FRAMES", sign, topConf
            )
        }

        // ── Cooldown ─────────────────────────────────────────────────────
        val now = SystemClock.elapsedRealtime()
        if (now - lastEmitAt < COOLDOWN_MS) {
            return Result.Rejected("cooldown", sign, topConf)
        }

        lastEmitAt = now
        stableCount = 0

        return Result.Detected(sign, topConf, elapsedMs)
    }

    /**
     * Standard deviation of each feature across the last [MOTION_WINDOW]
     * frames, averaged. Low means the hand is holding a shape.
     */
    private fun isStill(): Boolean {
        val recent = buffer.toList().takeLast(MOTION_WINDOW)
        if (recent.size < MOTION_WINDOW) return false

        var total = 0f
        for (f in 0 until FEATURE_DIM) {
            var mean = 0f
            for (frame in recent) mean += frame[f]
            mean /= recent.size

            var variance = 0f
            for (frame in recent) {
                val d = frame[f] - mean
                variance += d * d
            }
            total += sqrt(variance / recent.size)
        }

        return (total / FEATURE_DIM) < MOTION_THRESHOLD
    }

    /** Clear the sequence. Called when the hands leave the frame. */
    fun reset() {
        buffer.clear()
        stableSign = -1
        stableCount = 0
    }

    fun close() = interpreter.close()
}