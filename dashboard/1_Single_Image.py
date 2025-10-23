# pages/1_🖼️_Single_Image.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2 # Make sure OpenCV is installed (pip install opencv-python-headless)

# --- Configuration ---
MODEL_PATH = '../resnet50_v1.keras' # Path relative to this script
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ["AI GENERATED (FAKE)", "AUTHENTIC (REAL)"]

# --- Load Model (Cached) ---
@st.cache_resource
def load_verifyai_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}") # Also print to console
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

# --- Grad-CAM Logic (Manual Implementation) ---
def deprocess_resnet(x):
    x = x.copy()
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    x = x[..., ::-1] # BGR -> RGB
    x = np.clip(x, 0, 255) / 255.0
    return x

@st.cache_data # Cache Grad-CAM generation
def make_gradcam_heatmap(_model, img_array_preprocessed, last_conv_layer_name="conv5_block3_out"):
    try:
        last_conv_layer = _model.get_layer('resnet50').get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            _model.inputs, [last_conv_layer.output, _model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array_preprocessed)
            class_output = preds[:, 0]

        grads = tape.gradient(class_output, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10) # add epsilon
        return heatmap.numpy()
    except Exception as e:
        st.warning(f"Could not generate Grad-CAM: {e}")
        return None

def create_overlay_heatmap(heatmap, img_display_pil):
    """ Overlays the heatmap on the original image. """
    if heatmap is None:
        return None
    # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, (img_display_pil.width, img_display_pil.height))
    heatmap_jet = np.uint8(plt.cm.jet(heatmap_resized)[..., :3] * 255)
    heatmap_pil = Image.fromarray(heatmap_jet)

    # Ensure original image is RGB
    if img_display_pil.mode != "RGB":
        img_display_pil = img_display_pil.convert("RGB")

    # Blend original image and heatmap
    overlayed_img = Image.blend(img_display_pil, heatmap_pil, alpha=0.5)
    return overlayed_img

# --- Streamlit Page UI ---
st.header("🖼️ Single Image Classification")
st.markdown("Upload an image to check if it's AI-generated or authentic.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    key="single_uploader" # Unique key for this uploader
)

# Use session state to store results per upload
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "gradcam_heatmap" not in st.session_state:
    st.session_state.gradcam_heatmap = None
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "processed_image_key" not in st.session_state:
    st.session_state.processed_image_key = None


if uploaded_file is not None:
    # Check if it's a new file
    if uploaded_file.id != st.session_state.processed_image_key:
        # Reset previous results for the new file
        st.session_state.prediction = None
        st.session_state.confidence = None
        st.session_state.gradcam_heatmap = None
        st.session_state.original_image = None
        st.session_state.processed_image_key = uploaded_file.id

        image_bytes = uploaded_file.getvalue()
        preprocessed_image, img_display_pil = preprocess_image(image_bytes)

        if preprocessed_image is not None and model is not None:
            st.session_state.original_image = img_display_pil # Store original PIL image

            # Perform Prediction
            try:
                pred_value = model.predict(preprocessed_image)[0][0]
                if pred_value < 0.5:
                    st.session_state.prediction = CLASS_NAMES[0] # FAKE
                    st.session_state.confidence = (1 - pred_value) * 100
                else:
                    st.session_state.prediction = CLASS_NAMES[1] # REAL
                    st.session_state.confidence = pred_value * 100

                # Generate Grad-CAM only after prediction
                with st.spinner('Generating visualization...'):
                     st.session_state.gradcam_heatmap = make_gradcam_heatmap(model, preprocessed_image)

            except Exception as e:
                st.error(f"Error during prediction or Grad-CAM: {e}")
                # Clear potentially inconsistent state
                st.session_state.prediction = "Error"
                st.session_state.confidence = 0
                st.session_state.gradcam_heatmap = None
        else:
             # Clear state if preprocessing failed
            st.session_state.processed_image_key = None # Allow re-processing if needed


# --- Display Area ---
if st.session_state.original_image is not None:
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📸 Original Image", "🔥 Heatmap (Grad-CAM)", "📊 Results"])

    with tab1:
        st.image(st.session_state.original_image, caption="Uploaded Image", use_column_width="auto") # Use auto width

    with tab2:
        if st.session_state.gradcam_heatmap is not None:
            overlay_img = create_overlay_heatmap(st.session_state.gradcam_heatmap, st.session_state.original_image)
            if overlay_img:
                 st.image(overlay_img, caption="Grad-CAM Heatmap Overlay", use_column_width="auto")
                 st.caption("Warmer colors (red) indicate regions the model focused on.")
            else:
                st.warning("Could not display heatmap overlay.")
        elif st.session_state.prediction is not None and st.session_state.prediction != "Error":
             st.info("Generating heatmap...") # Show if prediction is done but heatmap isn't ready
        elif st.session_state.prediction == "Error":
             st.error("Cannot generate heatmap due to prediction error.")
        else:
             st.info("Heatmap will appear here after analysis.")


    with tab3:
        if st.session_state.prediction is not None and st.session_state.prediction != "Error":
            st.subheader("Analysis Result")
            col1, col2 = st.columns(2)
            with col1:
                 st.metric(label="Prediction", value=st.session_state.prediction)
            with col2:
                 st.metric(label="Confidence", value=f"{st.session_state.confidence:.2f}%")

            st.progress(int(st.session_state.confidence)) # Use int() here

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
        else:
            st.info("Results will appear here after analysis.")

elif uploaded_file is None:
     # Clear state if no file is uploaded
     st.session_state.prediction = None
     st.session_state.confidence = None
     st.session_state.gradcam_heatmap = None
     st.session_state.original_image = None
     st.session_state.processed_image_key = None