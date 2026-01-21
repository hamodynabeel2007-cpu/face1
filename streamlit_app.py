import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

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

def preprocess_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.reshape(face, (1, 48, 48, 1))

    return face, (x, y, w, h)

def predict_emotion(face_img):
    preds = model.predict(face_img, verbose=0)
    emotion = EMOTIONS[np.argmax(preds)]
    confidence = float(np.max(preds))
    return emotion, confidence

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

        face_img, box = preprocess_face(img_bgr)

        if face_img is None:
            st.error(“:x: No face detected.“)
        else:
            emotion, confidence = predict_emotion(face_img)
            st.success(f”Emotion: {emotion} ({confidence*100:.2f}%)“)

            x, y, w, h = box
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            st.image(img_rgb, caption=“Detected Face”, use_column_width=True)

# -------------------------
# Webcam Mode
# -------------------------
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format=“bgr24")

        face_img, box = preprocess_face(img)

        if face_img is not None:
            emotion, confidence = predict_emotion(face_img)
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
[8:35 PM] app.py
[8:35 PM] streamlit
tensorflow
opencv-python
pillow
streamlit-webrtc
numpy
[8:35 PM] requirements.txtimport streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

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

def preprocess_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.reshape(face, (1, 48, 48, 1))

    return face, (x, y, w, h)

def predict_emotion(face_img):
    preds = model.predict(face_img, verbose=0)
    emotion = EMOTIONS[np.argmax(preds)]
    confidence = float(np.max(preds))
    return emotion, confidence

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

        face_img, box = preprocess_face(img_bgr)

        if face_img is None:
            st.error(“:x: No face detected.“)
        else:
            emotion, confidence = predict_emotion(face_img)
            st.success(f”Emotion: {emotion} ({confidence*100:.2f}%)“)

            x, y, w, h = box
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            st.image(img_rgb, caption=“Detected Face”, use_column_width=True)

# -------------------------
# Webcam Mode
# -------------------------
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format=“bgr24")

        face_img, box = preprocess_face(img)

        if face_img is not None:
            emotion, confidence = predict_emotion(face_img)
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
[8:35 PM] app.py
[8:35 PM] streamlit
tensorflow
opencv-python
pillow
streamlit-webrtc
numpy
[8:35 PM] requirements.txtimport streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

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

def preprocess_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.reshape(face, (1, 48, 48, 1))

    return face, (x, y, w, h)

def predict_emotion(face_img):
    preds = model.predict(face_img, verbose=0)
    emotion = EMOTIONS[np.argmax(preds)]
    confidence = float(np.max(preds))
    return emotion, confidence

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

        face_img, box = preprocess_face(img_bgr)

        if face_img is None:
            st.error(“:x: No face detected.“)
        else:
            emotion, confidence = predict_emotion(face_img)
            st.success(f”Emotion: {emotion} ({confidence*100:.2f}%)“)

            x, y, w, h = box
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            st.image(img_rgb, caption=“Detected Face”, use_column_width=True)

# -------------------------
# Webcam Mode
# -------------------------
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format=“bgr24")

        face_img, box = preprocess_face(img)

        if face_img is not None:
            emotion, confidence = predict_emotion(face_img)
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
[8:35 PM] app.py
[8:35 PM] streamlit
tensorflow
opencv-python
pillow
streamlit-webrtc
numpy
[8:35 PM] requirements.txtimport streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

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

def preprocess_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.reshape(face, (1, 48, 48, 1))

    return face, (x, y, w, h)

def predict_emotion(face_img):
    preds = model.predict(face_img, verbose=0)
    emotion = EMOTIONS[np.argmax(preds)]
    confidence = float(np.max(preds))
    return emotion, confidence

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

        face_img, box = preprocess_face(img_bgr)

        if face_img is None:
            st.error(“:x: No face detected.“)
        else:
            emotion, confidence = predict_emotion(face_img)
            st.success(f”Emotion: {emotion} ({confidence*100:.2f}%)“)

            x, y, w, h = box
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            st.image(img_rgb, caption=“Detected Face”, use_column_width=True)

# -------------------------
# Webcam Mode
# -------------------------
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format=“bgr24")

        face_img, box = preprocess_face(img)

        if face_img is not None:
            emotion, confidence = predict_emotion(face_img)
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
[8:35 PM] app.py
[8:35 PM] streamlit
tensorflow
opencv-python
pillow
streamlit-webrtc
numpy
[8:35 PM] requirements.txtimport streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

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

def preprocess_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.reshape(face, (1, 48, 48, 1))

    return face, (x, y, w, h)

def predict_emotion(face_img):
    preds = model.predict(face_img, verbose=0)
    emotion = EMOTIONS[np.argmax(preds)]
    confidence = float(np.max(preds))
    return emotion, confidence

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

        face_img, box = preprocess_face(img_bgr)

        if face_img is None:
            st.error(“:x: No face detected.“)
        else:
            emotion, confidence = predict_emotion(face_img)
            st.success(f”Emotion: {emotion} ({confidence*100:.2f}%)“)

            x, y, w, h = box
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            st.image(img_rgb, caption=“Detected Face”, use_column_width=True)

# -------------------------
# Webcam Mode
# -------------------------
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format=“bgr24")

        face_img, box = preprocess_face(img)

        if face_img is not None:
            emotion, confidence = predict_emotion(face_img)
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
