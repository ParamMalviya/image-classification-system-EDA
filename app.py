import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the saved model
model = load_model('fruit_classification_model.h5')

# Get class names from model (update if needed)
class_names = ['Apple', 'avocado', 'Banana', 'cherry', 'kiwi', 'mango', 'orange', 'pinenapple', 'strawberries', 'watermelon']

# Streamlit app title
st.title("🍎🥝🍒 Fruit Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload a fruit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = load_img(uploaded_file, target_size=(128, 128))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Show prediction
    st.markdown(f"### 🥳 Predicted Class: **{predicted_class}**")
