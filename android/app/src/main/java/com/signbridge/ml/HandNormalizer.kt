package com.signbridge.ml

import kotlin.math.sqrt
object HandNormalizer {

    const val LANDMARKS = 21
    const val HAND_DIM = LANDMARKS * 3   // 63
    const val FEATURE_DIM = HAND_DIM * 2 // 126

    /** Matches the Python guard: max_d > 1e-6 */
    private const val EPSILON = 1e-6f

    /**
     * Normalize one hand's 63 values in place-safe fashion.
     *
     * 1. Translate so landmark 0 (wrist) sits at the origin
     * 2. Divide by the largest Euclidean distance from the wrist
     *
     * An all-zero input (hand not detected) returns all zeros rather than
     * NaN — the same behaviour as the Python guard. This matters because
     * most single-handed signs leave one half of the vector empty.
     */
    fun normalizeHand(hand63: FloatArray): FloatArray {
        require(hand63.size == HAND_DIM) {
            "Expected $HAND_DIM values, got ${hand63.size}"
        }

        val out = FloatArray(HAND_DIM)

        val wristX = hand63[0]
        val wristY = hand63[1]
        val wristZ = hand63[2]

        // Translate to wrist origin, tracking the max distance as we go
        var maxDist = 0f
        for (i in 0 until LANDMARKS) {
            val b = i * 3
            val x = hand63[b] - wristX
            val y = hand63[b + 1] - wristY
            val z = hand63[b + 2] - wristZ

            out[b] = x
            out[b + 1] = y
            out[b + 2] = z

            val d = sqrt(x * x + y * y + z * z)
            if (d > maxDist) maxDist = d
        }

        // Scale by hand size. Below epsilon the vector is already all zeros.
        if (maxDist > EPSILON) {
            for (i in out.indices) out[i] /= maxDist
        }

        return out
    }

    /**
     * Build the 126-dim model input from two hands.
     *
     * Pass a zero-filled FloatArray(63) for a hand that was not detected —
     * do NOT pass null-substituted garbage or skip the slot, since the model
     * was trained with zeros in that position.
     *
     * Hand order is [left, right] and is NOT interchangeable. Getting this
     * backwards produces a model that appears to work but confuses any sign
     * whose two hands differ.
     */
    fun buildFeatures(leftHand63: FloatArray, rightHand63: FloatArray): FloatArray {
        val features = FloatArray(FEATURE_DIM)
        normalizeHand(leftHand63).copyInto(features, 0)
        normalizeHand(rightHand63).copyInto(features, HAND_DIM)
        return features
    }

    /** Convenience: a not-detected hand. */
    fun emptyHand(): FloatArray = FloatArray(HAND_DIM)

    /**
     * Extract the hand portions of a raw 258-dim desktop keypoint vector.
     * Layout: [pose(132), leftHand(63), rightHand(63)].
     *
     * Used only for testing against desktop fixtures — on Android the
     * landmarks come straight from MediaPipe HandLandmarker, so there is no
     * pose block to skip.
     */
    fun fromRaw258(raw258: FloatArray): FloatArray {
        require(raw258.size == 258) { "Expected 258 values, got ${raw258.size}" }
        return buildFeatures(
            raw258.copyOfRange(132, 195),
            raw258.copyOfRange(195, 258)
        )
    }
}