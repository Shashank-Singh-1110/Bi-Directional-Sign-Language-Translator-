package com.signbridge.ml

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Verifies the Kotlin normalization against fixtures exported from the
 * Python implementation (export_test_fixtures.py).
 *
 * Place android_fixtures.json in app/src/test/resources/.
 *
 * Tolerance is 1e-5. Python computes in float64 and Kotlin in float32, so
 * exact equality is not achievable — but 1e-5 is roughly four orders of
 * magnitude tighter than anything that could affect a prediction.
 */
class HandNormalizerTest {

    private val tolerance = 1e-5f

    private fun loadFixtures(): JSONObject {
        val stream = javaClass.classLoader
            ?.getResourceAsStream("android_fixtures.json")
            ?: error("android_fixtures.json not found in test resources")
        return JSONObject(stream.bufferedReader().readText())
    }

    @Test
    fun matchesPythonOutputOnRealFrames() {
        val cases = loadFixtures().getJSONArray("cases")
        assertTrue("No fixtures loaded", cases.length() > 0)

        var worstDelta = 0f
        var worstCase = ""

        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val name = case.getString("name")

            val rawArr = case.getJSONArray("raw258")
            val raw = FloatArray(258) { rawArr.getDouble(it).toFloat() }

            val expectedArr = case.getJSONArray("expected126")
            val expected = FloatArray(126) { expectedArr.getDouble(it).toFloat() }

            val actual = HandNormalizer.fromRaw258(raw)

            assertEquals("$name: wrong output size", 126, actual.size)

            for (j in expected.indices) {
                val delta = abs(expected[j] - actual[j])
                if (delta > worstDelta) {
                    worstDelta = delta
                    worstCase = "$name[$j]"
                }
                assertTrue(
                    "$name index $j: expected ${expected[j]}, got ${actual[j]} " +
                            "(delta $delta)",
                    delta < tolerance
                )
            }
        }

        println("Checked ${cases.length()} fixtures. " +
                "Worst delta $worstDelta at $worstCase")
    }

    @Test
    fun undetectedHandStaysZero() {
        val result = HandNormalizer.normalizeHand(HandNormalizer.emptyHand())

        assertEquals(63, result.size)
        for (v in result) {
            assertTrue("Zero input produced $v — check the epsilon guard",
                v == 0f)
        }
    }

    @Test
    fun wristLandsAtOrigin() {
        // Arbitrary hand, offset well away from the origin
        val hand = FloatArray(63) { 0.5f + it * 0.01f }
        val result = HandNormalizer.normalizeHand(hand)

        assertTrue("Wrist x not at origin: ${result[0]}", abs(result[0]) < 1e-6f)
        assertTrue("Wrist y not at origin: ${result[1]}", abs(result[1]) < 1e-6f)
        assertTrue("Wrist z not at origin: ${result[2]}", abs(result[2]) < 1e-6f)
    }

    @Test
    fun furthestLandmarkIsUnitDistance() {
        val hand = FloatArray(63) { 0.3f + it * 0.013f }
        val result = HandNormalizer.normalizeHand(hand)

        var maxDist = 0f
        for (i in 0 until 21) {
            val b = i * 3
            val d = sqrt(
                result[b] * result[b] +
                        result[b + 1] * result[b + 1] +
                        result[b + 2] * result[b + 2]
            )
            if (d > maxDist) maxDist = d
        }

        assertTrue("Max distance after scaling was $maxDist, expected 1.0",
            abs(maxDist - 1f) < 1e-5f)
    }

    @Test
    fun scaleInvariance() {
        // The whole point of the normalization: the same hand shape at two
        // different distances from the camera must produce the same vector.
        val near = FloatArray(63) { 0.4f + it * 0.02f }
        val far = FloatArray(63) { near[it] * 0.35f }   // same shape, smaller

        val a = HandNormalizer.normalizeHand(near)
        val b = HandNormalizer.normalizeHand(far)

        var dot = 0f
        var normA = 0f
        var normB = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val cosine = dot / (sqrt(normA) * sqrt(normB))

        assertTrue("Cosine similarity across scales was $cosine, expected ~1.0",
            cosine > 0.999f)
    }

    @Test
    fun handOrderIsPreserved() {
        val left = FloatArray(63) { 0.2f + it * 0.01f }
        val right = FloatArray(63) { 0.8f - it * 0.01f }

        val features = HandNormalizer.buildFeatures(left, right)
        assertEquals(126, features.size)

        val expectedLeft = HandNormalizer.normalizeHand(left)
        val expectedRight = HandNormalizer.normalizeHand(right)

        for (i in 0 until 63) {
            assertEquals("Left hand corrupted at $i",
                expectedLeft[i], features[i], 1e-6f)
            assertEquals("Right hand corrupted at $i",
                expectedRight[i], features[63 + i], 1e-6f)
        }
    }
}