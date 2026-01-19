import cv2
import numpy as np
import os
import tensorflow as tf
from collections import deque
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import random

# ================= APP =================
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # sab domains allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# ================= MODEL =================
MODEL_PATH = r"C:\Ram Antivirus\Project\Liveness Detection\face_web_demo\backend\model\liveness_cnn_model.h5"

print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded")

CLASS_NAMES = [
    "live_video",
    "fake_photo",
    "fake_mask_video",
    "fake_deepfake_video"
]

IMG_SIZE = 128

# ================= OPENCV =================
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ================= SESSION =================
WINDOW = 15   # ~5 seconds

history = deque(maxlen=WINDOW)
motion_history = deque(maxlen=WINDOW)

prev_gray = None
blink_counter = 0
eye_visible_prev = True

challenge = random.choice(["BLINK", "MOVE_HEAD"])
benchmark = {"total": 0, "live": 0, "fake": 0}

# ================= UTILS =================

def reset_session():
    global prev_gray, blink_counter, eye_visible_prev, challenge
    history.clear()
    motion_history.clear()
    prev_gray = None
    blink_counter = 0
    eye_visible_prev = True
    challenge = random.choice(["BLINK", "MOVE_HEAD"])


def preprocess(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame.astype("float32") / 255.0
    return np.expand_dims(frame, axis=0)


def detect_motion(frame):
    global prev_gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        return 0.0

    diff = cv2.absdiff(prev_gray, gray)
    prev_gray = gray
    return float(np.mean(diff))


def detect_blink(gray):
    global blink_counter, eye_visible_prev
    eyes = eye_detector.detectMultiScale(gray, 1.3, 5)
    eye_visible_now = len(eyes) > 0

    if eye_visible_prev and not eye_visible_now:
        blink_counter += 1

    eye_visible_prev = eye_visible_now
    return blink_counter


def detect_screen_texture(gray):
    # High sharpness indicates mobile screen
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var


def predict_ai(frame):
    inp = preprocess(frame)
    preds = model.predict(inp, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx])

# ================= API =================

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    global benchmark, challenge

    img = await file.read()
    frame = cv2.imdecode(np.frombuffer(img, np.uint8), 1)

    if frame is None:
        return {"status": "error"}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- FACE CHECK ----
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return {"status": "blocked", "reason": "No face detected"}

    if len(faces) > 1:
        return {"status": "blocked", "reason": "Multiple faces detected"}

    # ---- SIGNALS ----
    motion = detect_motion(frame)
    blink = detect_blink(gray)
    texture_score = detect_screen_texture(gray)

    label, conf = predict_ai(frame)

    history.append(label)
    motion_history.append(motion)

    # ---- COLLECTING ----
    if len(history) < WINDOW:
        return {
            "status": "collecting",
            "frames": len(history),
            "challenge": challenge,
            "blink": blink
        }

    # ================= FINAL DECISION =================

    motion_avg = float(np.mean(motion_history))

    # ---- CHALLENGE VALIDATION ----
    challenge_passed = False

    if challenge == "BLINK" and blink >= 1:
        challenge_passed = True

    if challenge == "MOVE_HEAD" and motion_avg > 3.0:
        challenge_passed = True

    # ---- SCREEN TEXTURE SPOOF ----
    if texture_score > 1200:
        final_result = "FAKE"
        reason = "Mobile Screen Texture Detected"

    # ---- CHALLENGE FAILED ----
    elif not challenge_passed:
        final_result = "FAKE"
        reason = f"Challenge Failed ({challenge})"

    # ---- MODEL BASED FAKE ----
    elif label in ["fake_photo", "fake_mask_video", "fake_deepfake_video"]:
        final_result = "FAKE"
        reason = f"Model Detected: {label}"

    # ---- REAL FACE ----
    else:
        final_result = "VERIFIED"
        reason = "Live Human Verified"

    # ---- BENCHMARK ----
    benchmark["total"] += 1
    if final_result == "VERIFIED":
        benchmark["live"] += 1
    else:
        benchmark["fake"] += 1

    # ---- RESET ----
    reset_session()

    return {
        "status": "done",
        "final_result": final_result,
        "reason": reason,
        "challenge": challenge,
        "blink": blink,
        "motion_avg": round(motion_avg, 2),
        "texture_score": round(texture_score, 1),
        "benchmark": benchmark
    }

