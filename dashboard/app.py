# app.py (Streamlit Main Entry Point)
import streamlit as st

st.set_page_config(
    page_title="VerifyAI Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Configuration (Common to all pages) ---
with st.sidebar:
    # st.image("logo.png", width=150) # Optional: Add a logo image file
    st.title("VerifyAI Navigation")
    st.info("**Model:** ResNet-50 (Transfer Learning)\n\n**Accuracy:** ~96%")

    st.markdown("---")
    st.subheader("Project Info")
    st.markdown(
        "This tool detects AI-generated images vs. authentic photos "
        "using a deep learning model."
    )
    # Add more sidebar content common to all pages if needed

# --- Main Page Content (Home/About) ---
st.title("🔍 AI-Generated Image Detector")
st.markdown("### Welcome! Use the navigation sidebar to:")
st.markdown("- Classify a single image")
st.markdown("- Analyze batches of images (Coming Soon)")
st.markdown("- View model performance details (Coming Soon)")

st.markdown("---")
st.header("Project Overview")
st.write(
    "Deepfakes and AI-generated images are becoming increasingly sophisticated. "
    "VerifyAI aims to provide a tool to help distinguish between authentic photographs "
    "and images created by generative AI models like Stable Diffusion, Midjourney, etc."
)
# Add more content like sample images if desired here.