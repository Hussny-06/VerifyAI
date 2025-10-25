# app.py - VerifyAI v1 (ResNet-50)

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# 1. Initialize the Flask Application
app = Flask(__name__, static_folder='frontend', template_folder='frontend')

# 2. Define Image Size 
# Our ResNet-50 model expects 128x128 images
IMG_HEIGHT = 128
IMG_WIDTH = 128

# 3. Load the Trained Model at Startup 
print("Loading ResNet-50 deepfake detector, please wait...")
try:
    # Load the new, high-performance .keras model
    model = tf.keras.models.load_model('resnet50_v1.keras')
    print("Model resnet50_v1.keras loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# 4. Preprocessing Function (CRITICAL MODIFICATION)
# preprocessing similar to model training .
def preprocess_image(image_file):
    """
    Takes an image file, opens it, resizes it to 128x128,
    converts it to a NumPy array, and applies ResNet-50 preprocessing.
    """
    # Open the image file from the request
    img = Image.open(image_file.stream)
    
    # Resize the image
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Convert to RGB if it's not (e.g., PNG with alpha)
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    # The model expects a "batch" of images. Add a dimension.
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply the *specific* ResNet-50 preprocessing
    # This scales pixels to the format ImageNet was trained on.
    preprocessed_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    
    return preprocessed_array

# 5. Prediction Function 
def predict_deepfake(image_array):
    """
    Takes a preprocessed image array and returns a prediction from the model.
    """
    if model is None:
        return {"error": "Model is not loaded."}

    # Our model outputs 0 for FAKE and 1 for REAL
    prediction = model.predict(image_array)[0][0]
    
    if prediction < 0.5:
        # Prediction is closer to 0 (FAKE)
        confidence = 1 - prediction
        return {
            "prediction": "AI GENERATED (FAKE)",
            "confidence": f"{confidence:.2%}"
        }
    else:
        # Prediction is closer to 1 (REAL)
        confidence = prediction
        return {
            "prediction": "AUTHENTIC (REAL)",
            "confidence": f"{confidence:.2%}"
        }

# 6. Define the API Endpoint 
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

# 7. Define the Route for the Main Page 
@app.route('/')
def home():
    return render_template('index.html')

# 8. Run the Application 
if __name__ == '__main__':
    # threaded=False is often recommended when using TensorFlow with Flask
    # to avoid potential conflicts in multi-threaded environments.
    app.run(debug=True, threaded=False)