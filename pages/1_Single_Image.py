# pages/1_🖼️_Single_Image.py (Final - No Grad-CAM)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# --- Configuration ---
MODEL_PATH = 'resnet50_v1.keras' # Path relative to the root app.py
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ["AI GENERATED (FAKE)", "AUTHENTIC (REAL)"]

# --- Load Model (Cached) ---
@st.cache_resource
def load_verifyai_model(model_path):
    try:
        # Load model without compiling the optimizer state if it causes issues
        # (Compile=False is often safer for inference-only loading)
        model = tf.keras.models.load_model(model_path, compile=False) 
        print("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}")
        return None

model = load_verifyai_model(MODEL_PATH)

# --- Preprocessing ---
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img_display = img.copy() # Keep original for display
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        preprocessed_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        return preprocessed_array, img_display
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None

# --- Streamlit Page UI ---
st.header("🖼️ Single Image Classification")
st.markdown("Upload an image to check if it's AI-generated or authentic.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    key="single_uploader"
)

# Use session state for results
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "processed_file_key" not in st.session_state:
    st.session_state.processed_file_key = None


if uploaded_file is not None:
    current_file_key = f"{uploaded_file.name}-{uploaded_file.size}"

    # Only process if it's a new file
    if st.session_state.processed_file_key != current_file_key:
        st.session_state.prediction = None # Reset previous results
        st.session_state.confidence = None
        st.session_state.original_image = None
        st.session_state.processed_file_key = current_file_key

        image_bytes = uploaded_file.getvalue()
        preprocessed_image, img_display_pil = preprocess_image(image_bytes)

        if preprocessed_image is not None and model is not None:
            st.session_state.original_image = img_display_pil

            try:
                with st.spinner('Analyzing image...'):
                    pred_value = model.predict(preprocessed_image)[0][0]
                    if pred_value < 0.5:
                        st.session_state.prediction = CLASS_NAMES[0] # FAKE
                        st.session_state.confidence = (1 - pred_value) * 100
                    else:
                        st.session_state.prediction = CLASS_NAMES[1] # REAL
                        st.session_state.confidence = pred_value * 100
                st.success("Analysis Complete!")

            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.session_state.prediction = "Error"
                st.session_state.confidence = 0
        else:
            st.session_state.processed_file_key = None # Allow reprocessing if preprocessing failed

    # --- Display Area ---
    if st.session_state.original_image is not None:
        # Simplified: Just show image and results directly
        st.image(st.session_state.original_image, caption="Uploaded Image", use_column_width="auto")
        st.markdown("---") # Separator

        if st.session_state.prediction is not None and st.session_state.prediction != "Error":
            st.subheader("Analysis Result")
            col1, col2 = st.columns(2)
            with col1:
                 st.metric(label="Prediction", value=st.session_state.prediction)
            with col2:
                 st.metric(label="Confidence", value=f"{st.session_state.confidence:.2f}%")

            st.progress(int(st.session_state.confidence))

            # Optional: Confidence Breakdown (can keep this part)
            st.markdown("---")
            st.subheader("Confidence Breakdown")
            if st.session_state.prediction == CLASS_NAMES[0]: # FAKE
                 real_conf = 100 - st.session_state.confidence
                 fake_conf = st.session_state.confidence
            else: # REAL
                 real_conf = st.session_state.confidence
                 fake_conf = 100 - st.session_state.confidence

            col_real, col_fake = st.columns(2)
            with col_real:
                 st.write(f"**{CLASS_NAMES[1]}**") # REAL
                 st.progress(int(real_conf))
                 st.write(f"{real_conf:.2f}%")
            with col_fake:
                 st.write(f"**{CLASS_NAMES[0]}**") # FAKE
                 st.progress(int(fake_conf))
                 st.write(f"{fake_conf:.2f}%")

        elif st.session_state.prediction == "Error":
             st.error("Analysis failed. Please try a different image.")
        # No else needed, spinner shows while processing

else: # If uploaded_file is None
     if st.session_state.processed_file_key is not None:
        st.session_state.prediction = None
        st.session_state.confidence = None
        st.session_state.original_image = None
        st.session_state.processed_file_key = None
        # No rerun needed here, just clears state