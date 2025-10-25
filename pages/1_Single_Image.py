# pages/1_Single_Image.py (Final - No Grad-CAM)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
# NOTE: Removed cv2 and matplotlib imports as Grad-CAM was removed

# --- Configuration ---
MODEL_PATH = 'resnet50_v1.keras' # Path relative to the root app.py
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ["AI GENERATED (FAKE)", "AUTHENTIC (REAL)"]

# --- Load Model (Cached) ---
@st.cache_resource # Cache the model loading for efficiency
def load_verifyai_model(model_path):
    # ... (function remains the same) ...
    try:
        # Load model without compiling optimizer state (safer for inference)
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}")
        return None

model = load_verifyai_model(MODEL_PATH)

# --- Preprocessing ---
# (function remains the same)
def preprocess_image(image_bytes):
    # ... (function remains the same) ...
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img_display = img.copy() # Keep original PIL image for display
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        preprocessed_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        return preprocessed_array, img_display # Return both processed array and display image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None

# --- Streamlit Page UI ---
st.header("🖼️ Single Image Classification")
st.markdown("Upload an image to check if it's AI-generated or authentic.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    key="single_uploader" # Use a key to potentially differentiate uploaders later
)

# --- Session State Management ---
# Use session state to keep track of the current analysis results
# This prevents reprocessing when the user interacts with other elements
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "processed_file_key" not in st.session_state: # Key to track the currently processed file
    st.session_state.processed_file_key = None

# --- Main Processing Logic ---
if uploaded_file is not None:
    # Create a unique identifier for this specific uploaded file instance
    current_file_key = f"{uploaded_file.name}-{uploaded_file.size}"

    # Only re-process the image if it's a new file upload
    if st.session_state.processed_file_key != current_file_key:
        print(f"Processing new file: {uploaded_file.name}") # Log file processing
        st.session_state.prediction = None # Reset previous results
        st.session_state.confidence = None
        st.session_state.original_image = None
        st.session_state.processed_file_key = current_file_key # Mark this file as processed

        image_bytes = uploaded_file.getvalue()
        # Preprocess the image for the model and get the display version
        preprocessed_image, img_display_pil = preprocess_image(image_bytes)

        if preprocessed_image is not None and model is not None:
            # Store the displayable image in session state
            st.session_state.original_image = img_display_pil

            # Perform Prediction
            try:
                with st.spinner('Analyzing image...'): # Show spinner during prediction
                    pred_value = model.predict(preprocessed_image)[0][0]
                    # Determine class and confidence based on sigmoid output
                    if pred_value < 0.5:
                        st.session_state.prediction = CLASS_NAMES[0] # FAKE
                        st.session_state.confidence = (1 - pred_value) * 100
                    else:
                        st.session_state.prediction = CLASS_NAMES[1] # REAL
                        st.session_state.confidence = pred_value * 100
                st.success("Analysis Complete!") # Show success message

            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.session_state.prediction = "Error" # Mark state as error
                st.session_state.confidence = 0
        else:
            # If preprocessing failed, reset the key to allow reprocessing
            st.session_state.processed_file_key = None
            print("Preprocessing failed, resetting processed key.") # Log failure

    # --- Display Area ---
    # This section runs every time based on the current session state
    if st.session_state.original_image is not None:
        # Display the original uploaded image
        # Use use_container_width=True as recommended by the deprecation warning
        st.image(st.session_state.original_image, caption="Uploaded Image", use_container_width=True)
        st.markdown("---") # Separator

        # Display results if prediction was successful
        if st.session_state.prediction is not None and st.session_state.prediction != "Error":
            st.subheader("Analysis Result")
            col1, col2 = st.columns(2)
            with col1:
                 st.metric(label="Prediction", value=st.session_state.prediction)
            with col2:
                 st.metric(label="Confidence", value=f"{st.session_state.confidence:.2f}%")

            # Show progress bar based on confidence
            st.progress(int(st.session_state.confidence))

            # Display confidence breakdown
            st.markdown("---")
            st.subheader("Confidence Breakdown")
            # Calculate confidence for both classes
            if st.session_state.prediction == CLASS_NAMES[0]: # FAKE
                 real_conf = 100 - st.session_state.confidence
                 fake_conf = st.session_state.confidence
            else: # REAL
                 real_conf = st.session_state.confidence
                 fake_conf = 100 - st.session_state.confidence

            # Show breakdown using columns and progress bars
            col_real, col_fake = st.columns(2)
            with col_real:
                 st.write(f"**{CLASS_NAMES[1]}**") # REAL
                 st.progress(int(real_conf))
                 st.write(f"{real_conf:.2f}%")
            with col_fake:
                 st.write(f"**{CLASS_NAMES[0]}**") # FAKE
                 st.progress(int(fake_conf))
                 st.write(f"{fake_conf:.2f}%")

        # Display error message if analysis failed
        elif st.session_state.prediction == "Error":
             st.error("Analysis failed. Please try a different image.")
        # If processing is happening (e.g., spinner is active), this section won't show yet

# Handle case where user removes the uploaded file
elif uploaded_file is None:
     # If a file *was* processed but is now removed, clear the state
     if st.session_state.processed_file_key is not None:
        print("Clearing session state because file was removed.")
        st.session_state.prediction = None
        st.session_state.confidence = None
        st.session_state.original_image = None
        st.session_state.processed_file_key = None
        st.rerun() # Force a rerun to update the display immediately