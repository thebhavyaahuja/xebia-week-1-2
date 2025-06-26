import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Custom object to handle batch_shape compatibility
def custom_input_layer(*args, **kwargs):
    # Remove problematic batch_shape and replace with input_shape
    if 'batch_shape' in kwargs:
        batch_shape = kwargs.pop('batch_shape')
        if batch_shape and len(batch_shape) > 1:
            kwargs['input_shape'] = batch_shape[1:]
    return tf.keras.layers.InputLayer(*args, **kwargs)

# Custom DTypePolicy for compatibility
class DTypePolicy:
    def __init__(self, name='float32'):
        self.name = name
        self._compute_dtype = name
        self._variable_dtype = name
    
    @property
    def compute_dtype(self):
        return self._compute_dtype
    
    @property
    def variable_dtype(self):
        return self._variable_dtype

# Load model with custom objects
try:
    custom_objects = {
        'InputLayer': custom_input_layer,
        'DTypePolicy': DTypePolicy,
        'dtype_policy': DTypePolicy
    }
    model = tf.keras.models.load_model(
        'lung_cancer_model.h5', 
        custom_objects=custom_objects,
        compile=False
    )
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()
# streamlit page
st.set_page_config(page_title="Lung Cancer Prediction App", page_icon=":hospital:", layout="centered")
st.title("Lung Cancer Prediction App")
st.markdown("Paste the image to predict lung cancer:")

# Input fields for user data
image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
# Button to trigger prediction
if st.button("Predict"):
    if image is not None:
        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(image, target_size=(256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image

        # Make prediction
        prediction = model.predict(img_array)
        prediction_text = "Lung Cancer Detected" if prediction[0][0] > 0.5 else "No Lung Cancer Detected"
        
        st.success(f"The model predicts that the patient has: {prediction_text}")
    else:
        st.error("Please upload an image to make a prediction.")