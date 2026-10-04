package com.signbridge.ml

import android.content.Context
import android.os.SystemClock
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

/**
 * Wraps MediaPipe HandLandmarker and converts its output into the 126-dim
 * vector the sign model expects.
 *
 * Runs in LIVE_STREAM mode: detect() returns immediately and results arrive
 * on [listener], so the camera thread is never blocked.
 *
 * Three things have to be right for the vector to match training data:
 *
 *  1. ASPECT RATIO. MediaPipe normalizes x by image width and y by image
 *     height independently. The desktop pipeline captured 640x480 landscape;
 *     a portrait camera stream divides by the opposite dimensions, which
 *     stretches the hand by ~1.8x in one axis. Wrist-origin normalization
 *     removes position and scale but NOT anisotropic stretch, so this must
 *     be corrected before normalizing. See [trainingAspect].
 *
 *  2. WHICH SLOT. The feature vector is [leftHand(63), rightHand(63)] and
 *     nearly every sign here is one-handed, so the occupied half decides
 *     everything. See [slotMode].
 *
 *  3. MISSING HANDS contribute 63 zeros, never anything else.
 */
class HandLandmarkerHelper(
    private val context: Context,
    private val listener: Listener,
    private val minDetectionConfidence: Float = 0.5f,
    private val minTrackingConfidence: Float = 0.5f,
    private val minPresenceConfidence: Float = 0.5f,
    private val useGpu: Boolean = false,
) {

    /** Where a detected hand is placed in the feature vector. */
    enum class Slot {
        /** Trust MediaPipe's handedness label. */
        AUTO,

        /** Force into the first 63 values. */
        LEFT,

        /** Force into the last 63 values. */
        RIGHT,
    }

    /**
     * Aspect ratio (width / height) of the frames the model was trained on.
     * The desktop pipeline used OpenCV's default webcam capture, 640x480.
     */
    var trainingAspect: Float = 640f / 480f

    /** Correct for the capture aspect ratio before normalizing. */
    var correctAspect: Boolean = true

    var slotMode: Slot = Slot.AUTO

    /** Invert MediaPipe's Left/Right labels. Only used when [slotMode] is AUTO. */
    var swapHandedness: Boolean = false

    /**
     * Extra in-plane rotation applied to the landmarks, in degrees.
     *
     * A rotation sweep against the training data found the emulator feed is
     * offset by about 45 degrees: correcting it lifted the similarity to the
     * correct class from 0.58 to 0.78. On real hardware this is likely to be
     * 0, since the sensor rotation is handled by rotating the frame itself.
     */
    var extraRotationDeg: Float = 45f

    interface Listener {
        fun onResults(
            features: FloatArray,
            handCount: Int,
            inferenceMs: Long,
            result: HandLandmarkerResult,
        )

        fun onError(message: String)
    }

    private var landmarker: HandLandmarker? = null

    /** Dimensions of the most recently submitted frame, for aspect correction. */
    @Volatile private var frameWidth = 0
    @Volatile private var frameHeight = 0

    companion object {
        private const val MODEL_ASSET = "hand_landmarker.task"
        private const val MAX_HANDS = 2
        private const val LANDMARKS = 21
    }

    init {
        setup()
    }

    private fun setup() {
        try {
            val base = BaseOptions.builder()
                .setModelAssetPath(MODEL_ASSET)
                .setDelegate(if (useGpu) Delegate.GPU else Delegate.CPU)
                .build()

            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(base)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumHands(MAX_HANDS)
                .setMinHandDetectionConfidence(minDetectionConfidence)
                .setMinTrackingConfidence(minTrackingConfidence)
                .setMinHandPresenceConfidence(minPresenceConfidence)
                .setResultListener { result, _ -> handleResult(result) }
                .setErrorListener { e ->
                    listener.onError(e.message ?: "MediaPipe error")
                }
                .build()

            landmarker = HandLandmarker.createFromOptions(context, options)
        } catch (e: Exception) {
            listener.onError(
                "Failed to initialise HandLandmarker: ${e.message}. " +
                        "Check that $MODEL_ASSET is in src/main/assets."
            )
        }
    }

    /**
     * Submit a frame. Returns immediately; results arrive on the listener.
     *
     * @param timestampMs must increase monotonically — MediaPipe silently
     *        drops frames whose timestamp is not newer than the last.
     */
    fun detect(image: MPImage, timestampMs: Long) {
        frameWidth = image.width
        frameHeight = image.height
        landmarker?.detectAsync(image, timestampMs)
    }

    private var lastStart = 0L

    private fun handleResult(result: HandLandmarkerResult) {
        val elapsed = SystemClock.uptimeMillis() - lastStart
        lastStart = SystemClock.uptimeMillis()

        val (features, count) = pack(result)
        listener.onResults(features, count, elapsed, result)
    }

    /**
     * Convert MediaPipe output into the model's 126-dim input.
     *
     * Layout: [leftHand(63), rightHand(63)], 21 landmarks x (x, y, z) each,
     * every hand normalized independently to its own wrist and scale.
     */
    private fun pack(result: HandLandmarkerResult): Pair<FloatArray, Int> {
        var left: FloatArray? = null
        var right: FloatArray? = null

        val hands = result.landmarks()
        val labels = result.handedness()

        // Undo the camera's aspect ratio and re-apply the training one, so
        // the hand has the same proportions the model learned.
        //
        // MediaPipe gives x in units of 1/width and y in units of 1/height.
        // Multiplying by the real dimensions returns true (isotropic) pixel
        // shape; dividing by the training dimensions reintroduces exactly
        // the distortion present in the training data.
        var sx = 1f
        var sy = 1f
        if (correctAspect && frameWidth > 0 && frameHeight > 0) {
            val cameraAspect = frameWidth.toFloat() / frameHeight.toFloat()
            // Only the ratio between the axes matters — overall scale is
            // removed by the wrist normalization anyway.
            sx = cameraAspect / trainingAspect
        }

        // In-plane rotation correction, applied about the wrist so it
        // commutes with the normalization that follows.
        val rad = Math.toRadians(extraRotationDeg.toDouble())
        val cos = kotlin.math.cos(rad).toFloat()
        val sin = kotlin.math.sin(rad).toFloat()
        val rotating = extraRotationDeg != 0f

        for (i in hands.indices) {
            val raw = FloatArray(HandNormalizer.HAND_DIM)
            val points = hands[i]

            val wristX = points[0].x() * sx
            val wristY = points[0].y() * sy

            for (j in 0 until LANDMARKS) {
                val lm = points[j]
                var x = lm.x() * sx
                var y = lm.y() * sy

                if (rotating) {
                    val dx = x - wristX
                    val dy = y - wristY
                    x = wristX + (cos * dx - sin * dy)
                    y = wristY + (sin * dx + cos * dy)
                }

                raw[j * 3] = x
                raw[j * 3 + 1] = y
                raw[j * 3 + 2] = lm.z() * sx
            }

            val goesLeft = when (slotMode) {
                Slot.LEFT -> true
                Slot.RIGHT -> false
                Slot.AUTO -> {
                    val label = labels.getOrNull(i)
                        ?.firstOrNull()
                        ?.categoryName()
                        ?: continue
                    if (swapHandedness) label == "Right" else label == "Left"
                }
            }

            // With a forced slot and two hands visible, the second would
            // overwrite the first. Keep the first and ignore the rest.
            if (goesLeft) {
                if (left == null) left = raw
            } else {
                if (right == null) right = raw
            }
        }

        val count = listOfNotNull(left, right).size

        val features = HandNormalizer.buildFeatures(
            left ?: HandNormalizer.emptyHand(),
            right ?: HandNormalizer.emptyHand(),
        )

        return features to count
    }

    fun close() {
        landmarker?.close()
        landmarker = null
    }
}