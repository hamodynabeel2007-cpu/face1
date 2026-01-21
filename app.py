"""
Facial Emotion Detection Application
A Streamlit app for real-time emotion detection from images and webcam.
"""

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import logging
from typing import Optional, Tuple, List
from pathlib import Path

# -------------------------
# Configuration
# -------------------------
MODEL_PATH = "model (1).h5"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
INPUT_SIZE = (48, 48)
MAX_FILE_SIZE_MB = 10
CONFIDENCE_THRESHOLD = 0.3
PROCESS_EVERY_N_FRAMES = 2  # For webcam performance

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Facial Emotion Detector",
    page_icon="🎭",
    layout="centered"
)

# -------------------------
# Model Loading with Error Handling
# -------------------------
@st.cache_resource
def load_model(model_path: str) -> Optional[tf.keras.Model]:
    """
    Load the trained emotion detection model.

    Args:
        model_path: Path to the saved model file

    Returns:
        Loaded Keras model or None if loading fails
    """
    try:
        if not Path(model_path).exists():
            st.error(f"❌ Model file not found: {model_path}")
            st.info("Please ensure the model file is in the working directory.")
            return None

        model = tf.keras.models.load_model(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        logger.error(f"Model loading error: {e}")
        return None


@st.cache_resource
def load_face_cascade(cascade_path: str) -> Optional[cv2.CascadeClassifier]:
    """
    Load Haar Cascade classifier for face detection.

    Args:
        cascade_path: Path to the cascade XML file

    Returns:
        Loaded cascade classifier or None if loading fails
    """
    try:
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            st.error("❌ Failed to load face cascade classifier.")
            return None
        logger.info("Successfully loaded face cascade classifier")
        return cascade
    except Exception as e:
        st.error(f"❌ Error loading cascade: {str(e)}")
        logger.error(f"Cascade loading error: {e}")
        return None


# Load resources
model = load_model(MODEL_PATH)
face_cascade = load_face_cascade(FACE_CASCADE_PATH)

# Stop if critical resources failed to load
if model is None or face_cascade is None:
    st.stop()

# -------------------------
# Face Processing Functions
# -------------------------
def preprocess_face(img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Detect and preprocess face from image.

    Args:
        img: Input image in BGR format

    Returns:
        Tuple of (preprocessed_face, bounding_box) or (None, None) if no face found
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)  # Minimum face size
        )

        if len(faces) == 0:
            return None, None

        # Process the largest face (most prominent)
        # Sort by area and take the largest
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces_sorted[0]

        # Extract and preprocess face region
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, INPUT_SIZE)
        face = face / 255.0  # Normalize
        face = np.reshape(face, (1, INPUT_SIZE[0], INPUT_SIZE[1], 1))

        return face, (x, y, w, h)

    except Exception as e:
        logger.error(f"Face preprocessing error: {e}")
        return None, None


def preprocess_all_faces(img: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Detect and preprocess all faces in image.

    Args:
        img: Input image in BGR format

    Returns:
        Tuple of (list_of_preprocessed_faces, list_of_bounding_boxes)
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

        processed_faces = []
        boxes = []

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, INPUT_SIZE)
            face = face / 255.0
            face = np.reshape(face, (1, INPUT_SIZE[0], INPUT_SIZE[1], 1))

            processed_faces.append(face)
            boxes.append((x, y, w, h))

        return processed_faces, boxes

    except Exception as e:
        logger.error(f"Multi-face preprocessing error: {e}")
        return [], []


def predict_emotion(face_img: np.ndarray) -> Tuple[str, float]:
    """
    Predict emotion from preprocessed face image.

    Args:
        face_img: Preprocessed face array

    Returns:
        Tuple of (emotion_label, confidence_score)
    """
    try:
        preds = model.predict(face_img, verbose=0)
        emotion_idx = np.argmax(preds)
        emotion = EMOTIONS[emotion_idx]
        confidence = float(np.max(preds))

        return emotion, confidence

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Unknown", 0.0


def validate_image_upload(uploaded_file) -> bool:
    """
    Validate uploaded image file.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        True if valid, False otherwise
    """
    if uploaded_file is None:
        return False

    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"❌ File too large. Maximum size: {MAX_FILE_SIZE_MB}MB")
        return False

    return True


# -------------------------
# WebRTC Video Processor
# -------------------------
class EmotionProcessor(VideoProcessorBase):
    """
    Real-time video processor for emotion detection.
    Optimized for performance with frame skipping.
    """

    def __init__(self):
        self.frame_count = 0
        self.last_emotion = "Processing..."
        self.last_confidence = 0.0
        self.last_box = None

    def recv(self, frame):
        """
        Process incoming video frame.

        Args:
            frame: Input video frame from WebRTC

        Returns:
            Processed video frame with annotations
        """
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")

        # Process every Nth frame for performance
        self.frame_count += 1
        if self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
            face_img, box = preprocess_face(img)

            if face_img is not None:
                emotion, confidence = predict_emotion(face_img)

                # Only update if confidence is above threshold
                if confidence >= CONFIDENCE_THRESHOLD:
                    self.last_emotion = emotion
                    self.last_confidence = confidence
                    self.last_box = box

        # Draw annotations using last known values
        if self.last_box is not None:
            x, y, w, h = self.last_box

            # Draw rectangle
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Draw label with background
            label = f"{self.last_emotion} ({self.last_confidence*100:.1f}%)"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            # Background rectangle for text
            cv2.rectangle(
                img,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                (0, 255, 0),
                -1
            )

            # Text
            cv2.putText(
                img,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -------------------------
# Main Application UI
# -------------------------
st.title("🎭 Facial Emotion Detection")
st.markdown("Detect emotions from uploaded images or real-time webcam feed.")

# Mode selection
mode = st.radio(
    "Choose Input Method:",
    ["📁 Upload Image", "📹 Webcam"],
    horizontal=True
)

# -------------------------
# Upload Image Mode
# -------------------------
if mode == "📁 Upload Image":
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=["jpg", "png", "jpeg"],
        help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB"
    )

    detect_all_faces = st.checkbox(
        "Detect all faces in image",
        value=False,
        help="When enabled, detects and analyzes all faces. Otherwise, only the largest face."
    )

    if uploaded_file and validate_image_upload(uploaded_file):
        try:
            # Load and convert image
            image = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Process based on mode
            if detect_all_faces:
                faces, boxes = preprocess_all_faces(img_bgr)

                if len(faces) == 0:
                    st.error("❌ No faces detected in the image.")
                else:
                    st.success(f"✅ Detected {len(faces)} face(s)")

                    # Process each face
                    for idx, (face_img, box) in enumerate(zip(faces, boxes)):
                        emotion, confidence = predict_emotion(face_img)

                        # Draw annotations
                        x, y, w, h = box
                        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        label = f"{emotion} ({confidence*100:.1f}%)"
                        cv2.putText(
                            img_bgr,
                            label,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                        # Display results
                        st.markdown(f"**Face {idx + 1}:** {emotion} ({confidence*100:.2f}%)")

            else:
                # Single face mode
                face_img, box = preprocess_face(img_bgr)

                if face_img is None:
                    st.error("❌ No face detected in the image.")
                    st.info("💡 Try using 'Detect all faces' mode or ensure the face is clearly visible.")
                else:
                    emotion, confidence = predict_emotion(face_img)

                    # Draw annotations
                    x, y, w, h = box
                    cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    label = f"{emotion} ({confidence*100:.1f}%)"
                    cv2.putText(
                        img_bgr,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )

                    # Display results
                    st.success(f"**Emotion:** {emotion} | **Confidence:** {confidence*100:.2f}%")

            # Display annotated image
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Detection Result", use_column_width=True)

        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            logger.error(f"Image processing error: {e}")

# -------------------------
# Webcam Mode
# -------------------------
elif mode == "📹 Webcam":
    st.markdown("---")
    st.info("📸 Allow webcam access to start real-time emotion detection.")
    st.markdown(f"**Note:** Processing every {PROCESS_EVERY_N_FRAMES} frames for optimal performance.")

    webrtc_streamer(
        key="emotion-webcam",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <small>Built with Streamlit, TensorFlow, and OpenCV</small>
    </div>
    """,
    unsafe_allow_html=True
)
