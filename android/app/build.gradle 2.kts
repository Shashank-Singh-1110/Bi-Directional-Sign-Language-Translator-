    plugins {
        alias(libs.plugins.android.application)
        alias(libs.plugins.kotlin.compose)
    }

    android {
        namespace = "com.signbridge"
        compileSdk {
            version = release(37)
        }

        defaultConfig {
            applicationId = "com.signbridge"
            minSdk = 26
            targetSdk = 37
            versionCode = 1
            versionName = "1.0"

            testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        }

        buildTypes {
            release {
                optimization {
                    enable = true
                    packageScope = setOf("androidx.**", "kotlin.**", "kotlinx.**")
                }
            }
        }
        compileOptions {
            sourceCompatibility = JavaVersion.VERSION_11
            targetCompatibility = JavaVersion.VERSION_11
        }
        buildFeatures {
            compose = true
        }
        androidResources{
            noCompress += "tflite"
        }
    }

    dependencies {
        implementation(platform(libs.androidx.compose.bom))
        implementation(libs.androidx.activity.compose)
        implementation(libs.androidx.compose.material3)
        implementation(libs.androidx.compose.ui)
        implementation(libs.androidx.compose.ui.graphics)
        implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.4")
        implementation(libs.androidx.compose.ui.tooling.preview)
        implementation(libs.androidx.core.ktx)
        implementation(libs.androidx.lifecycle.runtime.ktx)
        testImplementation(libs.junit)
        androidTestImplementation(platform(libs.androidx.compose.bom))
        androidTestImplementation(libs.androidx.compose.ui.test.junit4)
        androidTestImplementation(libs.androidx.espresso.core)
        androidTestImplementation(libs.androidx.junit)
        debugImplementation(libs.androidx.compose.ui.test.manifest)
        debugImplementation(libs.androidx.compose.ui.tooling)
        implementation("com.google.mediapipe:tasks-vision:0.10.14")
        implementation("org.tensorflow:tensorflow-lite:2.16.1")
        implementation("androidx.camera:camera-core:1.3.4")
        implementation("androidx.camera:camera-camera2:1.3.4")
        implementation("androidx.camera:camera-lifecycle:1.3.4")
        implementation("androidx.camera:camera-view:1.3.4")
        testImplementation("junit:junit:4.13.2")
        testImplementation("org.json:json:20240303")
    }