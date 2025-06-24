# streamlit_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pathlib

# Load class names (manually define or load from dataset if available)
class_names = ['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon', 'hamburger',
               'ice_cream', 'pizza', 'ramen', 'steak', 'sushi']

# Load trained model
model = tf.keras.models.load_model("saved_train_model.h5")

# Image preprocessing function
def load_and_prep_image(img, img_shape=224):
    img = img.resize((img_shape, img_shape))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, axis=0)
    img = img / 255.
    return img

# Streamlit UI
st.title("🍔 Food Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = load_and_prep_image(image)

    prediction = model.predict(img)
    pred_class = class_names[np.argmax(prediction)]

    st.markdown(f"### 🍽 Prediction: **{pred_class}**")
