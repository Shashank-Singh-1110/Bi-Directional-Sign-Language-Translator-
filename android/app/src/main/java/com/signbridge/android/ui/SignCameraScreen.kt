package com.signbridge.ui

import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.signbridge.ml.HandLandmarkerHelper
import com.signbridge.ml.SignClassifier
import java.util.concurrent.Executors

private const val FEATURE_TAG = "FEATURES"

/**
 * Live sign recognition screen.
 *
 * Tap "dump" to log the current 126-dim feature vector to Logcat under the
 * tag FEATURES. Capture it with:
 *
 *     adb logcat -d -s FEATURES | tail -1 > android_vector.txt
 *
 * then run scripts/convert/diagnose_android.py against it to find which
 * geometric transform reconciles Android's landmarks with the training data.
 *
 * The other chips vary the four things that can independently break that
 * match: sensor rotation, which half of the vector the hand occupies,
 * horizontal mirroring, and aspect-ratio correction.
 */
@Composable
fun SignCameraScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var detectedSign by remember { mutableStateOf("—") }
    var confidence by remember { mutableFloatStateOf(0f) }
    var status by remember { mutableStateOf("starting") }
    var handCount by remember { mutableIntStateOf(0) }
    var mediapipeMs by remember { mutableLongStateOf(0L) }
    var modelMs by remember { mutableLongStateOf(0L) }
    var history by remember { mutableStateOf(listOf<String>()) }
    var frameInfo by remember { mutableStateOf("") }
    var dumpNote by remember { mutableStateOf("") }

    var applyRotation by remember { mutableStateOf(false) }
    var slot by remember { mutableStateOf(HandLandmarkerHelper.Slot.AUTO) }
    var mirror by remember { mutableStateOf(false) }
    var aspect by remember { mutableStateOf(true) }

    // Set by the dump chip; the next frame with a hand writes to Logcat.
    val dumpRequested = remember { mutableStateOf(false) }

    val classifier = remember { SignClassifier(context) }
    val executor = remember { Executors.newSingleThreadExecutor() }

    val helper = remember {
        HandLandmarkerHelper(
            context = context,
            listener = object : HandLandmarkerHelper.Listener {
                override fun onResults(
                    features: FloatArray,
                    handCount: Int,
                    inferenceMs: Long,
                    result: HandLandmarkerResult,
                ) {
                    val hands = handCount
                    mediapipeMs = inferenceMs

                    if (dumpRequested.value && hands > 0) {
                        dumpRequested.value = false
                        Log.d(FEATURE_TAG, features.joinToString(","))
                        dumpNote = "dumped ${features.size} values to Logcat"
                    }

                    when (val r = classifier.addFrame(features, hands)) {
                        is SignClassifier.Result.Warming -> {
                            status = "buffering ${r.framesHeld}/30"
                        }

                        is SignClassifier.Result.Rejected -> {
                            status = r.reason
                            if (r.topSign.isNotEmpty()) {
                                detectedSign = r.topSign
                                confidence = r.confidence
                            }
                        }

                        is SignClassifier.Result.Detected -> {
                            status = "detected"
                            detectedSign = r.sign
                            confidence = r.confidence
                            modelMs = r.inferenceMs
                            history = (history + r.sign).takeLast(10)
                            // Also dump on every detection, so a vector is
                            // available even without tapping.
                            Log.d(FEATURE_TAG, features.joinToString(","))
                        }
                    }

                    reportHands(hands)
                }

                override fun onError(message: String) {
                    status = "error: $message"
                }

                /** Separate method so the parameter name does not shadow. */
                private fun reportHands(n: Int) {
                    handCount = n
                }
            },
        )
    }

    // Push settings into the helper and clear the window, so frames captured
    // under the old settings do not linger in the 30-frame buffer.
    LaunchedEffect(slot, aspect, mirror, applyRotation) {
        helper.slotMode = slot
        helper.correctAspect = aspect
        classifier.reset()
        history = emptyList()
        detectedSign = "—"
        confidence = 0f
        dumpNote = ""
    }

    // The analyzer runs on a background thread and needs current values,
    // not the ones captured when the lambda was created.
    val mirrorState = rememberUpdatedState(mirror)
    val rotationState = rememberUpdatedState(applyRotation)

    DisposableEffect(Unit) {
        onDispose {
            helper.close()
            classifier.close()
            executor.shutdown()
        }
    }

    Box(Modifier.fillMaxSize().background(Color.Black)) {

        AndroidView(
            factory = { ctx ->
                val previewView = PreviewView(ctx).apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                }

                val providerFuture = ProcessCameraProvider.getInstance(ctx)
                providerFuture.addListener({
                    val provider = providerFuture.get()

                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                    val analysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(
                            ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                        )
                        .setOutputImageFormat(
                            ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
                        )
                        .build()

                    analysis.setAnalyzer(executor) { imageProxy ->
                        try {
                            val degrees = imageProxy.imageInfo.rotationDegrees

                            val frame = transform(
                                imageProxy.toBitmap(),
                                if (rotationState.value) degrees.toFloat() else 0f,
                                mirrorState.value,
                            )

                            frameInfo = "${frame.width}x${frame.height} " +
                                    "(sensor r$degrees)"

                            helper.detect(
                                BitmapImageBuilder(frame).build(),
                                // Must increase monotonically or MediaPipe
                                // silently drops the frame.
                                imageProxy.imageInfo.timestamp / 1_000_000,
                            )
                        } catch (e: Exception) {
                            status = "frame error: ${e.message}"
                        } finally {
                            imageProxy.close()
                        }
                    }

                    provider.unbindAll()
                    provider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_FRONT_CAMERA,
                        preview,
                        analysis,
                    )
                }, androidx.core.content.ContextCompat.getMainExecutor(ctx))

                previewView
            },
            modifier = Modifier.fillMaxSize(),
        )

        Row(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Chip("dump", dumpRequested.value) { dumpRequested.value = true }
            Chip("rot", applyRotation) { applyRotation = !applyRotation }

            Chip(
                label = when (slot) {
                    HandLandmarkerHelper.Slot.AUTO -> "auto"
                    HandLandmarkerHelper.Slot.LEFT -> "L"
                    HandLandmarkerHelper.Slot.RIGHT -> "R"
                },
                active = slot != HandLandmarkerHelper.Slot.AUTO,
            ) {
                slot = when (slot) {
                    HandLandmarkerHelper.Slot.AUTO -> HandLandmarkerHelper.Slot.LEFT
                    HandLandmarkerHelper.Slot.LEFT -> HandLandmarkerHelper.Slot.RIGHT
                    HandLandmarkerHelper.Slot.RIGHT -> HandLandmarkerHelper.Slot.AUTO
                }
            }

            Chip("mir", mirror) { mirror = !mirror }
            Chip("asp", aspect) { aspect = !aspect }
        }

        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(Color.Black.copy(alpha = 0.78f))
                .padding(20.dp),
        ) {
            Text(
                text = detectedSign,
                color = Color.White,
                fontSize = 44.sp,
                fontWeight = FontWeight.Bold,
            )

            Text(
                text = if (confidence > 0) "${"%.1f".format(confidence)}%" else "",
                color = Color(0xFFC9A84C),
                fontSize = 18.sp,
            )

            Spacer(Modifier.height(12.dp))

            Text(
                text = "$status · ${handCount}h · mp ${mediapipeMs}ms · " +
                        "model ${modelMs}ms",
                color = Color.Gray,
                fontSize = 13.sp,
                fontFamily = FontFamily.Monospace,
            )

            Text(
                text = frameInfo,
                color = Color.Gray.copy(alpha = 0.7f),
                fontSize = 12.sp,
                fontFamily = FontFamily.Monospace,
            )

            if (dumpNote.isNotEmpty()) {
                Text(
                    text = dumpNote,
                    color = Color(0xFFC9A84C),
                    fontSize = 12.sp,
                    fontFamily = FontFamily.Monospace,
                )
            }

            if (history.isNotEmpty()) {
                Spacer(Modifier.height(8.dp))
                Text(
                    text = history.joinToString(" "),
                    color = Color.White.copy(alpha = 0.7f),
                    fontSize = 16.sp,
                )
            }
        }
    }
}

@Composable
private fun Chip(label: String, active: Boolean, onClick: () -> Unit) {
    Text(
        text = label,
        color = if (active) Color.Black else Color.White,
        fontSize = 12.sp,
        fontFamily = FontFamily.Monospace,
        modifier = Modifier
            .clip(RoundedCornerShape(6.dp))
            .background(
                if (active) Color(0xFFC9A84C) else Color.Black.copy(alpha = 0.6f)
            )
            .clickable(onClick = onClick)
            .padding(horizontal = 10.dp, vertical = 7.dp),
    )
}

/** Rotate upright, and optionally mirror, before detection. */
private fun transform(bitmap: Bitmap, degrees: Float, mirror: Boolean): Bitmap {
    if (degrees == 0f && !mirror) return bitmap

    val matrix = Matrix().apply {
        if (degrees != 0f) postRotate(degrees)
        if (mirror) postScale(-1f, 1f)
    }

    return Bitmap.createBitmap(
        bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
    )
}