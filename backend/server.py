import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define absolute paths based on this script's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'smart_leaf_doctor_mobilenetv2.h5')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'models', 'class_names.json')

# Global variables for model and class names
model = None
class_names = []

def load_resources():
    global model, class_names
    if os.path.exists(MODEL_PATH):
        try:
            print("Loading Model...")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model file not found at {MODEL_PATH}")

    if os.path.exists(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, 'r') as f:
                class_names = json.load(f)
            print("Class names loaded successfully.")
        except Exception as e:
            print(f"Error loading class names: {e}")
    else:
        print(f"Class names file not found at {CLASS_NAMES_PATH}")

# Load resources at startup
load_resources()

def get_recommendation(is_healthy, disease_name):
    if is_healthy:
        return "Excellent! The plant tissue structure appears perfectly normal. Maintain current care routine with proper hydration and light."
    
    disease_lower = disease_name.lower()
    
    if "blight" in disease_lower:
        return "Apply a copper-based fungicide immediately. Isolate affected plants to prevent spore dispersion. Reduce overhead watering to lower humidity around the foliage canopy."
    elif "rust" in disease_lower:
        return "Remove and destroy infected leaves. Apply a sulfur-based fungicide. Ensure good air circulation around the plants."
    elif "spot" in disease_lower:
        return "Treat with appropriate broad-spectrum fungicide. Avoid wetting leaves during watering. Remove decaying debris around the base."
    elif "virus" in disease_lower or "mosaic" in disease_lower:
        return "Viruses cannot be cured. Immediately remove and destroy the infected plant to prevent spread via insects. Control local aphid populations."
    elif "mildew" in disease_lower:
        return "Improve air circulation. Apply neem oil or a sulfur fungicide. Avoid watering late in the day to allow leaves to dry."
    else:
        return f"Infection detected. It is highly recommended to isolate the plant and apply appropriate targeted treatment for '{disease_name}'."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded. Please ensure training is complete."}), 503
        
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read image
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        
        # Convert to RGB if RGBA (e.g. some PNGs)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Preprocess
        img_resized = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Predict
        predictions = model.predict(img_array)[0]
        pred_idx = np.argmax(predictions)
        
        if len(class_names) > 0:
            predicted_class = class_names[pred_idx]
        else:
            predicted_class = f"Class_Index_{pred_idx}"
            
        confidence = float(predictions[pred_idx] * 100)
        
        # Clean up name (e.g., "Tomato___Early_blight" -> "Early blight")
        parts = predicted_class.split('___')
        plant_name = parts[0].replace('_', ' ')
        disease_name = parts[-1].replace('_', ' ') if len(parts) > 1 else plant_name
        
        display_name = f"{plant_name} - {disease_name}"
        
        is_healthy = "healthy" in predicted_class.lower()
        status = "Healthy" if is_healthy else "Diseased"
        recommendation = get_recommendation(is_healthy, disease_name)
        
        # We can also generate a base64 of the image to display on the frontend if needed, 
        # but the frontend already has the file locally.
        
        return jsonify({
            "success": True,
            "raw_class": predicted_class,
            "display_name": display_name,
            "disease_name": disease_name,
            "plant_name": plant_name,
            "confidence": confidence,
            "status": status,
            "is_healthy": is_healthy,
            "recommendation": recommendation
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    outputs_dir = os.path.join(BASE_DIR, 'outputs')
    return send_from_directory(outputs_dir, filename)

if __name__ == '__main__':
    print("Starting Flask server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
