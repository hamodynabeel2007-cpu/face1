import streamlit as st
mport streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image


@st.cache_resource
def load_fer_model():
return load_model("your_model_name.h5")

model = load_fer_model()
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


st.title("Facial Emotion Recognition App")
st.write("Upload an image to see the detected emotion!")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

image = Image.open(uploaded_file)
st.image(image, caption='Uploaded Image', use_column_width=True)


img_array = np.array(image.convert('RGB'))
gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

# Face Detection (Crucial for FER2013)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
roi_gray = gray[y:y+h, x:x+w]
roi_gray = cv2.resize(roi_gray, (48, 48))
roi_gray = roi_gray.astype('float32') / 255.0
roi_gray = np.expand_dims(roi_gray, axis=0)
roi_gray = np.expand_dims(roi_gray, axis=-1)

# Prediction (Step G)
prediction = model.predict(roi_gray)
label = labels[np.argmax(prediction)]

st.success(f"The detected emotion is: **{label}**")
