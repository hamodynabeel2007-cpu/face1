import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf


st.set_page_config(
    page_title="Image Neural Network App",
    layout="centered"
)

st.title("🧠 Image Neural Network Demo")
st.write("Upload an image and let the neural network predict!")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model (1).h5")

model = load_model()


IMG_SIZE = 48  

CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

def preprocess_image(image):
    image = image.convert("L")          
    image = image.resize((48, 48))       
    img_array = np.array(image) / 255.0  
    img_array = np.expand_dims(img_array, axis=-1)  
    img_array = np.expand_dims(img_array, axis=0)   
    return img_array


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Model is thinking... 🤔"):
            img_array = preprocess_image(image)
            predictions = model.predict(img_array)

            confidence = np.max(predictions)
            predicted_class = CLASS_NAMES[np.argmax(predictions)]

        st.success(f"### Prediction: **{predicted_class}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")

       
        st.subheader("Class Probabilities")
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{class_name}: {predictions[0][i] * 100:.2f}%")
