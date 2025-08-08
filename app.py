# app.py

from flask import Flask, request, jsonify, render_template
import random # We'll use this to simulate the model's prediction

# 1. Initialize the Flask Application
# =====================================
# We create an instance of the Flask class. 
# 'static_folder' and 'template_folder' tell Flask where to find our frontend files.
app = Flask(__name__, static_folder='frontend', template_folder='frontend')


# 2. Define the Model Loading and Prediction (Placeholder)
# ========================================================
# In the future, this is where you will load your real model file (e.g., model.h5).
# For now, we create a placeholder function that simulates a prediction.
def predict_deepfake(image_file):
    """
    This is a placeholder function for our deepfake detection model.
    It takes an image file (which we ignore for now) and returns a fake prediction.
    
    In the future, this function will:
    1. Preprocess the image (resize, normalize).
    2. Pass it to the loaded TensorFlow/Keras model.
    3. Return the model's actual prediction.
    """
    print("Model simulation: Analyzing the image...")
    
    # Simulate a prediction
    is_fake = random.choice([True, False])
    confidence = random.uniform(0.75, 0.99)
    
    if is_fake:
        return {
            "prediction": "FAKE",
            "confidence": f"{confidence:.2%}" # Formats the number as a percentage
        }
    else:
        return {
            "prediction": "REAL",
            "confidence": f"{confidence:.2%}"
        }


# 3. Define the API Endpoint
# ==========================
# This is the URL that our JavaScript will send the image to.
# We specify `methods=['POST']` because the frontend will be sending data (the image).
@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    This function is triggered when a POST request is made to the /analyze URL.
    It handles the image upload, calls the model for a prediction, and returns the result.
    """
    # Check if an image file was sent in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    # Check if the user submitted an empty part without a filename
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
        
    if file:
        # We have the file! Now we send it to our placeholder model function.
        # The 'file' object itself is passed. You don't need to save it to disk yet.
        result = predict_deepfake(file)
        
        # Return the result from our model as a JSON response.
        # The frontend JavaScript will receive this JSON.
        return jsonify(result)

# 4. Define the Route for the Main Page
# =====================================
# This tells Flask to serve our `index.html` file when someone visits the root URL.
@app.route('/')
def home():
    """
    Serves the main HTML page of the application.
    """
    return render_template('index.html')


# 5. Run the Application
# ======================
# This block checks if the script is being run directly (not imported).
# If so, it starts the Flask development server.
# `debug=True` enables auto-reloading when you save changes.
if __name__ == '__main__':
    app.run(debug=True)

