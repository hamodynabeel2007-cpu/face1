import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import logging
from typing import Optional, Tuple, List
from pathlib import Path
import plotly.graph_objects as go

# -------------------------
# Configuration & Constants
# -------------------------
MODEL_PATH = "model (1).h5"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
INPUT_SIZE = (48, 48)

# STUN servers are required for WebRTC to work over the internet (outside localhost)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="Emotion AI", page_icon="🎭", layout="wide")

# -------------------------
# Sidebar - Settings
# -------------------------
st.sidebar.title("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
process_n_frames = st.sidebar.slider("Frame Skipping (Webcam)", 1, 10, 2)
st.sidebar.info("Higher 'Frame Skipping' improves performance on slower devices.")

# -------------------------
# Resource Loading
# -------------------------
@st.cache_resource
def load_model(path: str):
    try:
        if not Path(path).exists():
            return None
        return tf.keras.models.load_model(path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(FACE_CASCADE_PATH)

model = load_model(MODEL_PATH)
face_cascade = load_cascade()

# -------------------------
# Core Logic Functions
# -------------------------
def get_prediction(face_img: np.ndarray) -> np.ndarray:
    """Returns the full probability distribution."""
    preds = model.predict(face_img, verbose=0)
    return preds[0]

def plot_emotion_probs(probs):
    """Creates a horizontal bar chart of probabilities."""
    fig = go.Figure(go.Bar(
        x=probs,
        y=EMOTIONS,
        orientation='h',
        marker_color='skyblue'
    ))
    fig.update_layout(
        title="Emotion Probability Distribution",
        xaxis=dict(range=[0, 1]),
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

# -------------------------
# WebRTC Video Processor
# -------------------------
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_results = [] # Store (box, label, confidence)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % process_n_frames == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            new_results = []
            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, INPUT_SIZE) / 255.0
                roi = np.reshape(roi, (1, INPUT_SIZE[0], INPUT_SIZE[1], 1))
                
                preds = model.predict(roi, verbose=0)[0]
                idx = np.argmax(preds)
                conf = preds[idx]
                
                if conf >= confidence_threshold:
                    new_results.append(((x, y, w, h), EMOTIONS[idx], conf))
            self.last_results = new_results

        # Draw detections
        for (box, label, conf) in self.last_results:
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{label} ({conf:.2f})", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------
# Main UI
# -------------------------
if model is None:
    st.error(f"Model file '{MODEL_PATH}' not found. Please upload it to the directory.")
    st.stop()

st.title("🎭 Facial Emotion Detector")

tab1, tab2 = st.tabs(["📁 Image Upload", "📹 Real-time Webcam"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                st.warning("No faces detected.")
                st.image(image)
            else:
                # Analyze the first face for the chart
                x, y, w, h = faces[0]
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, INPUT_SIZE) / 255.0
                roi = np.reshape(roi, (1, INPUT_SIZE[0], INPUT_SIZE[1], 1))
                
                probs = get_prediction(roi)
                top_emotion = EMOTIONS[np.argmax(probs)]
                
                # Draw on image
                cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 4)
                st.image(img_array, caption=f"Detected: {top_emotion}")

    with col2:
        if uploaded_file and len(faces) > 0:
            st.subheader("Analysis")
            st.plotly_chart(plot_emotion_probs(probs), use_container_width=True)
            st.metric("Primary Emotion", top_emotion, f"{np.max(probs)*100:.1f}% Confidence")

with tab2:
    st.info("Ensure you have granted camera permissions in your browser.")
    webrtc_streamer(
        key="emotion-detection",
        mode=None,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

st.markdown("---")
st.caption("Powered by TensorFlow & Streamlit | Model Input: 48x48 Grayscale")
