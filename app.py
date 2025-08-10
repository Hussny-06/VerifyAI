# app.py - Final Version with Real Model

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image # PIL is from the Pillow library
import io # Used to handle the image file in memory

# 1. Initialize the Flask Application
# =====================================
app = Flask(__name__, static_folder='frontend', template_folder='frontend')


# 2. Load the Trained Model at Startup ### NEW ###
# =================================================
# We load the model only once when the application starts.
# This is the "strategic" way to do it for performance.
print("Loading the deepfake detection model, please wait...")
try:
    model = tf.keras.models.load_model('deepfake_detector_v1.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# 3. Preprocessing Function ### NEW ###
# =====================================
# This function prepares the user's image to match the input format of our model.
def preprocess_image(image_file):
    """
    Takes an image file, opens it, resizes it to 128x128,
    and converts it to a NumPy array that the model can understand.
    """
    # Open the image file from the request
    img = Image.open(image_file.stream)
    # Resize the image to the size our model expects (128x128)
    img = img.resize((128, 128))
    # Convert the image to a NumPy array
    img_array = np.array(img)
    # The model expects a "batch" of images. We add a dimension to create a batch of 1.
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# 4. Prediction Function ### MODIFIED ###
# ========================================
# This function now uses the real model to make a prediction.
def predict_deepfake(image_array):
    """
    Takes a preprocessed image array and returns a prediction from the model.
    """
    if model is None:
        return {"error": "Model is not loaded."}

    # Use the model to predict. The output will be a value between 0 and 1.
    prediction = model.predict(image_array)[0][0]
    
    # We'll set a threshold of 0.5 to decide between REAL and FAKE
    if prediction < 0.5:
        # The model's output for 'fake' was 0
        confidence = 1 - prediction
        return {
            "prediction": "FAKE",
            "confidence": f"{confidence:.2%}"
        }
    else:
        # The model's output for 'real' was 1
        confidence = prediction
        return {
            "prediction": "REAL",
            "confidence": f"{confidence:.2%}"
        }


# 5. Define the API Endpoint ### MODIFIED ###
# ==========================================
@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
        
    if file:
        try:
            # Preprocess the image for the model
            preprocessed_image = preprocess_image(file)
            
            # Get the prediction from our model
            result = predict_deepfake(preprocessed_image)
            
            return jsonify(result)
        except Exception as e:
            print(f"An error occurred during analysis: {e}")
            return jsonify({"error": "Failed to analyze image."}), 500


# 6. Define the Route for the Main Page (Unchanged)
# =================================================
@app.route('/')
def home():
    return render_template('index.html')


# 7. Run the Application (Unchanged)
# ==================================
if __name__ == '__main__':
    # The 'threaded=False' is important for some TensorFlow versions
    # to avoid issues with making predictions in a multi-threaded environment.
    app.run(debug=True, threaded=False)
