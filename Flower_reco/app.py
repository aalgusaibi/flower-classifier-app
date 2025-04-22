import os
import streamlit as st
import tensorflow as tf
import numpy as np
from keras.models import load_model

# Streamlit header
st.header('Flower Classification CNN Model')

# Class names
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load the pre-trained model with custom objects for augmentation layers
try:
    model = load_model(
        'Flower_Recog_Model.h5',
        custom_objects={
            "RandomZoom": tf.keras.layers.RandomZoom,
            "RandomFlip": tf.keras.layers.RandomFlip,
            "RandomRotation": tf.keras.layers.RandomRotation,
            "RandomContrast": tf.keras.layers.RandomContrast
        }
    )
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to classify images with threshold
def classify_images(image):
    input_image = tf.image.resize(image, (180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    confidence = np.max(result) * 100

    if confidence >= 80:
        outcome = (
            f"The Image belongs to **{flower_names[np.argmax(result)]}** "
            f"with a score of **{confidence:.2f}%**"
        )
    else:
        outcome = "The image is **NOT** confidently detected as a flower (confidence below 80%)."
    return outcome

# Option to choose input method
option = st.radio('Choose input method:', ('Upload an Image', 'Take a Photo'))

if option == 'Upload an Image':
    uploaded_file = st.file_uploader('Upload an Image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = tf.keras.utils.load_img(uploaded_file)
        st.image(uploaded_file, width=200)
        result = classify_images(image)
        st.write(result)

elif option == 'Take a Photo':
    picture = st.camera_input("Take a picture")
    if picture:
        image = tf.keras.utils.load_img(picture)
        st.image(picture, width=200)
        result = classify_images(image)
        st.write(result)
