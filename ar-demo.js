// Import MediaPipe Tasks-Vision FaceLandmarker
import {
    FaceLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const cameraBtn = document.getElementById("cameraBtn");
const statusText = document.getElementById("status");
const jewelryUpload = document.getElementById("jewelryUpload");
const jewelryType = document.getElementById("jewelryType");

// We need an image overlay - loading it programmatically
const earringImg = document.getElementById("earring_img");

const presetImages = document.getElementById("presetImages");

const presets = {
    earring: ["images/earring1.png", "images/earring2.png"],
    necklace: ["images/necklace1.png", "images/necklace2.png"],
    bracelet: ["images/brac1.png", "images/brac2.png"]
};

// Function to update preset images based on jewelry type
function updatePresetImages() {
    const type = jewelryType.value;
    presetImages.innerHTML = "";
    if (presets[type]) {
        presets[type].forEach(src => {
            const img = document.createElement("img");
            img.src = src;
            img.className = "preset-img";
            img.addEventListener("click", () => {
                // Remove selected class from all
                document.querySelectorAll(".preset-img").forEach(el => el.classList.remove("selected"));
                img.classList.add("selected");
                earringImg.src = src;
            });
            presetImages.appendChild(img);
        });
    }
}

// Initial call
updatePresetImages();

// Update on type change
jewelryType.addEventListener("change", () => {
    updatePresetImages();
});

// Add an event listener to handle custom user uploads
jewelryUpload.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
        // Deselect presets when uploading custom
        document.querySelectorAll(".preset-img").forEach(el => el.classList.remove("selected"));
        earringImg.src = URL.createObjectURL(file);
    }
});

let faceLandmarker;
let webcamRunning = false;
let lastVideoTime = -1;

// Dimensions
const videoWidth = 640;
const videoHeight = 480;

// Step 1: Initialize the FaceLandmarker API
async function initializeFaceLandmarker() {
    // FilesetResolver helps load the WebAssembly (WASM) models needed to run the AI in browser
    const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );

    // Create the landmarker
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU" // Uses WebGL for hardware acceleration
        },
        outputFaceBlendshapes: true,
        runningMode: "VIDEO", // Optimized for live camera feeds
        numFaces: 1 // Assuming 1 person is trying on jewelry at a time
    });

    // Model is downloaded and ready!
    statusText.innerText = "Model Loaded. Ready to turn on camera.";
    cameraBtn.disabled = false;
    cameraBtn.innerText = "Turn On Camera";
}

// Call initialization immediately
initializeFaceLandmarker();

// Step 2: Handle Camera activation
async function enableCam() {
    if (!faceLandmarker) {
        console.log("Wait for faceLandmarker to load before clicking!");
        return;
    }

    if (webcamRunning) {
        // Toggle off for simplicity (optional)
        webcamRunning = false;
        return;
    }

    const constraints = {
        video: {
            width: videoWidth,
            height: videoHeight
        }
    };

    try {
        // Request webcam access from the user
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;

        // Wait until video data is loaded before trying to predict
        video.addEventListener("loadeddata", predictWebcam);

        webcamRunning = true;
        cameraBtn.style.display = "none"; // Hide button once started
        statusText.innerText = "Camera active. Analyzing face...";

        // Ensure canvas perfectly aligns with video
        canvasElement.width = videoWidth;
        canvasElement.height = videoHeight;

    } catch (err) {
        statusText.innerText = "Error accessing camera: " + err;
        console.error(err);
    }
}

// Bind the button event
cameraBtn.addEventListener("click", enableCam);

// Step 3: Run real-time predictions and draw
async function predictWebcam() {
    if (!webcamRunning) return;

    let startTimeMs = performance.now();

    // Only detect if video has advanced to a new frame
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;

        // This is the core magic: detecting the 468 facial points
        const results = faceLandmarker.detectForVideo(video, startTimeMs);

        // Clear previous frame drawings
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        // If faces were detected
        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            statusText.innerText = "Face Detected! Adjusting Jewelry...";
            drawEarrings(results.faceLandmarks);
            // Hint for future: you can also draw necklaces by calculating points below the chin
        } else {
            statusText.innerText = "Please look at the camera...";
        }
    }

    // Call this function again for the next video frame
    window.requestAnimationFrame(predictWebcam);
}

// Step 4: Map the jewelry to specific facial landmarks
function drawEarrings(faceLandmarksArray) {
    // Array index 0 is the first face found
    const landmarks = faceLandmarksArray[0];
    const type = jewelryType.value;

    // Head proximity scaling heuristic based on eye distance
    const leftEye = landmarks[33];
    const rightEye = landmarks[263];
    const eyeDist = Math.abs((rightEye.x - leftEye.x) * canvasElement.width);
    const scaleFactor = eyeDist / 120; // 120 is average pixel distance at normal sitting range

    if (type === "earring") {
        /*
           MediaPipe Face Mesh Index Reference:
           - 132 or 177: Left Lower Earlobe
           - 361 or 401: Right Lower Earlobe
        */
        const leftEarlobe = landmarks[132];
        const rightEarlobe = landmarks[361];

        if (leftEarlobe && rightEarlobe) {
            const lx = leftEarlobe.x * canvasElement.width;
            const ly = leftEarlobe.y * canvasElement.height;
            const rx = rightEarlobe.x * canvasElement.width;
            const ry = rightEarlobe.y * canvasElement.height;

            const baseSize = 80;
            const finalSize = baseSize * scaleFactor;

            // Draw left earring
            canvasCtx.drawImage(earringImg, lx - (finalSize / 2), ly, finalSize, finalSize);
            // Draw right earring
            canvasCtx.drawImage(earringImg, rx - (finalSize / 2), ry, finalSize, finalSize);
        }
    } else if (type === "necklace") {
        // Point 152 is the bottom of the chin
        const chin = landmarks[152];
        if (chin) {
            const cx = chin.x * canvasElement.width;
            // Place the necklace slightly below the chin
            const cy = (chin.y * canvasElement.height) + (50 * scaleFactor);

            const baseSize = 250;
            const finalSize = baseSize * scaleFactor;

            // Draw necklace centered on the chest
            canvasCtx.drawImage(earringImg, cx - (finalSize / 2), cy, finalSize, finalSize);
        }
    }
}
