import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from collections import deque
import matplotlib.pyplot as plt

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title=“Facial Emotion Detector”, layout=“centered”)
st.title(“:performing_arts: Facial Emotion Detection”)

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(“emotion_model.h5")

model = load_model()

EMOTIONS = [“Angry”, “Disgust”, “Fear”, “Happy”, “Sad”, “Surprise”, “Neutral”]

# -------------------------
# Face Detection
# -------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + “haarcascade_frontalface_default.xml”
)

def preprocess_face(img, padding=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    processed_faces = []
    boxes = []

    for (x, y, w, h) in faces:
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)

        face = gray[y1:y2, x1:x2]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        processed_faces.append(face)
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    return processed_faces, boxes

def predict_emotions(faces):
    results = []
    for face_img in faces:
        preds = model.predict(face_img, verbose=0)[0]
        results.append(preds)
    return results

# -------------------------
# Emotion Smoothing Buffer
# -------------------------
SMOOTHING_WINDOW = 7

class EmotionSmoother:
    def __init__(self):
        self.buffers = {}

    def update(self, face_id, probs):
        if face_id not in self.buffers:
            self.buffers[face_id] = deque(maxlen=SMOOTHING_WINDOW)
        self.buffers[face_id].append(probs)
        return np.mean(self.buffers[face_id], axis=0)

smoother = EmotionSmoother()

# -------------------------
# UI Mode Selector
# -------------------------
mode = st.radio(“Choose Input Method:“, [“Upload Image”, “Webcam”])

# -------------------------
# Upload Image Mode
# -------------------------
if mode == “Upload Image”:
    uploaded_file = st.file_uploader(
        “Upload a face image”, type=[“jpg”, “png”, “jpeg”]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert(“RGB”)
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        faces, boxes = preprocess_face(img_bgr)

        if not faces:
            st.error(“:x: No faces detected.“)
        else:
            probs_list = predict_emotions(faces)

            for i, (probs, box) in enumerate(zip(probs_list, boxes)):
                emotion = EMOTIONS[np.argmax(probs)]
                confidence = float(np.max(probs))

                x, y, w, h = box
                cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f”{emotion} ({confidence*100:.1f}%)”
                cv2.putText(
                    img_bgr, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )

                st.subheader(f”Face {i+1}: {emotion}“)
                fig, ax = plt.subplots()
                ax.bar(EMOTIONS, probs)
                ax.set_ylim(0, 1)
                ax.set_ylabel(“Probability”)
                st.pyplot(fig)

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption=“Detected Faces”, use_column_width=True)

# -------------------------
# Webcam Mode
# -------------------------
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format=“bgr24")

        faces, boxes = preprocess_face(img)

        probs_list = predict_emotions(faces)

        for i, (probs, box) in enumerate(zip(probs_list, boxes)):
            smoothed_probs = smoother.update(i, probs)

            emotion = EMOTIONS[np.argmax(smoothed_probs)]
            confidence = float(np.max(smoothed_probs))

            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f”{emotion} ({confidence*100:.1f}%)”
            cv2.putText(
                img, label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        return img

if mode == “Webcam”:
    st.info(“:camera_with_flash: Allow webcam access to start detection.“)

    webrtc_streamer(
        key=“emotion-webcam”,
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={“video”: True, “audio”: False},
        async_processing=True,
    )
