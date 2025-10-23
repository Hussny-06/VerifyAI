# dashboard.py - Streamlit Interface for VerifyAI

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# --- Configuration ---
MODEL_PATH = 'resnet50_v1.keras' # Path to your champion model
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ["AI GENERATED (FAKE)", "AUTHENTIC (REAL)"]

# --- Load Model (Cached for performance) ---
@st.cache_resource # Use Streamlit's caching for the model
def load_verifyai_model(model_path):
    """Loads the trained Keras model."""
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_verifyai_model(MODEL_PATH)

# --- Preprocessing Function ---
def preprocess_image(image_bytes):
    """
    Preprocesses the uploaded image bytes for ResNet-50.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        preprocessed_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        return preprocessed_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# --- Streamlit App UI ---
st.set_page_config(page_title="VerifyAI Dashboard", layout="wide")

# Header
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://img.icons8.com/plasticine/100/000000/shield.png", width=100) # Simple shield icon
with col2:
    st.title("VerifyAI - Deepfake Image Detector")
    st.markdown("Upload an image to check if it's AI-generated or authentic.")

st.markdown("---")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption='Uploaded Image', use_column_width=True)

    # Preprocess and Predict
    with st.spinner('Analyzing image...'):
        preprocessed_image = preprocess_image(image_bytes)
        
        if preprocessed_image is not None:
            try:
                prediction = model.predict(preprocessed_image)[0][0]
                
                # Determine class and confidence
                if prediction < 0.5:
                    predicted_class = CLASS_NAMES[0] # FAKE
                    confidence = (1 - prediction) * 100
                else:
                    predicted_class = CLASS_NAMES[1] # REAL
                    confidence = prediction * 100
                    
                st.success(f"Analysis Complete!")
                
                # Display results
                st.subheader("Analysis Result:")
                st.metric(label="Prediction", value=predicted_class)
                st.metric(label="Confidence", value=f"{confidence:.2f}%")
                
                # Confidence bar (optional visualization)
                st.progress(int(confidence))
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")

elif model is None:
    st.error("Model could not be loaded. Please check the model file and logs.")

# Footer / Info
st.markdown("---")
st.sidebar.header("About VerifyAI")
st.sidebar.info(
    "This tool uses a ResNet-50 model trained on the CIFAKE dataset "
    "to classify images as either authentic or AI-generated. "
    "Model Accuracy: ~96%"
)
st.sidebar.header("How It Works")
st.sidebar.markdown("""
1.  **Upload:** Select an image file (JPG, JPEG, PNG).
2.  **Preprocess:** The image is resized and prepared for the model.
3.  **Analyze:** The ResNet-50 model predicts the probability.
4.  **Result:** The dashboard displays the classification (FAKE/REAL) and confidence level.
""")