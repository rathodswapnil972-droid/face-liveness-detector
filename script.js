const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const statusBox = document.getElementById("status");
const resultBox = document.getElementById("result");

let running = true;
let frameCounter = 0;
const MAX_FRAMES = 15;   // ~5 seconds

// ================= START CAMERA =================
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    statusBox.innerText = "Camera started. Scanning...";
    setTimeout(captureLoop, 800);   // small warmup delay
  } catch (err) {
    alert("❌ Camera access denied");
    console.error(err);
  }
}

// ================= CAPTURE LOOP =================
function captureLoop() {
  if (!running) return;

  frameCounter++;

  statusBox.innerText = `⏳ Scanning (${frameCounter}/${MAX_FRAMES})`;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);

  canvas.toBlob(blob => {
    sendFrame(blob);
  }, "image/jpeg");

  // ✅ IMPORTANT FIX
  if (frameCounter < MAX_FRAMES) {
    setTimeout(captureLoop, 300);
  }
}

// ================= SEND FRAME =================
function sendFrame(blob) {
  const formData = new FormData();
  formData.append("file", blob, "frame.jpg");

  fetch("/analyze", {
    method: "POST",
    body: formData
  })
  .then(res => res.json())
  .then(data => {

    console.log("Backend:", data);

    // Show challenge while scanning
    if (data.status === "collecting") {
      if (data.challenge) {
        statusBox.innerText =
          `⏳ Scanning (${data.frames}/15) | Challenge: ${data.challenge}`;
      }
    }

    // ✅ FINAL RESULT
    if (data.status === "done") {
      running = false;
      showResult(data.final_result);
    }

    // Blocked case
    if (data.status === "blocked") {
      running = false;
      statusBox.innerText = "🚫 BLOCKED";
      resultBox.innerText = data.reason;
    }

  })
  .catch(err => {
    console.error("Server error:", err);
    statusBox.innerText = "❌ Backend error";
  });
}

// ================= DISPLAY RESULT =================
function showResult(result) {

  statusBox.innerText = "STATUS: Scan Complete";

  if (result === "VERIFIED") {
    resultBox.innerText = "RESULT: VERIFIED";
    resultBox.className = "live";
  } else {
    resultBox.innerText = "RESULT: FAKE";
    resultBox.className = "fake";
    alert("⚠ FAKE FACE DETECTED!");
  }
}

// ================= AUTO START =================
startCamera();
