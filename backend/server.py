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

def get_recommendation_details(is_healthy, disease_name):
    disease_lower = disease_name.lower()
    
    # Default Image Fallbacks
    img_healthy = "https://images.unsplash.com/photo-1416879572648-52243d467fbd?w=400&q=80" # Fresh leaf
    img_blight = "https://images.unsplash.com/photo-1622383563227-04401ab4e5ea?w=400&q=80" # Decaying/dry leaf
    img_rust = "https://images.unsplash.com/photo-1597848212624-a19eb35e2651?w=400&q=80" # Brown/rust spots
    img_spot = "https://images.unsplash.com/photo-1615800098779-1be32e60cca3?w=400&q=80" # Spots
    img_virus = "https://images.unsplash.com/photo-1574868516008-01d0ed4b2a8f?w=400&q=80" # Yellowing/Mosaic
    img_general = "https://images.unsplash.com/photo-1587824859663-d1446abafbe8?w=400&q=80" # Laboratory/sick plant

    if is_healthy:
        return {
            "ar_title": "نبات سليم (حالة صحية ممتازة)",
            "ar_desc": "تشير التحليلات الدقيقة للأنسجة الخلوية إلى خلو النبات من أي أعراض فطرية أو بكتيرية أو فيروسية. البناء الضوئي يعمل بكفاءة قصوى.",
            "ar_treatment": "استمر في جدول الري المعتاد مع الحفاظ على معدلات التسميد الحالية. يرجى ضمان تعرض النبات لأشعة شمس كافية وتهوية جيدة.",
            "image": img_healthy
        }
    
    if "blight" in disease_lower:
        return {
            "ar_title": "اللفحة الفطرية (Blight)",
            "ar_desc": "هذا المرض الفطري الشرس يهاجم الأنسجة الحية للنبات بسرعة، مما يؤدي إلى ظهور بقع بنية داكنة وجفاف مفاجئ للأوراق، وقد يقضي على المحصول بالكامل إذا لم يُعالج.",
            "ar_treatment": "١. رش مبيد فطري يحتوي على النحاس فوراً.\n٢. إزالة الأوراق والسيقان المصابة وإبعادها.\n٣. تجنب الري الرذاذي لتقليل الرطوبة حول الأوراق.",
            "image": img_blight
        }
    elif "rust" in disease_lower:
        return {
            "ar_title": "صدأ الأوراق (Rust)",
            "ar_desc": "يتميز بظهور بثرات أو بقع تشبه الصدأ على السطح السفلي للأوراق. الفطر يسحب الغذاء من النبات ويضعف نموه بشكل حاد.",
            "ar_treatment": "١. تخلص من الأجزاء المصابة بحذر كي لا تتطاير الأبواغ.\n٢. استخدم مبيدات فطرية تحتوي على الكبريت.\n٣. وسّع المسافات بين النباتات لضمان تهوية ممتازة.",
            "image": img_rust
        }
    elif "spot" in disease_lower:
        return {
            "ar_title": "تبقع الأوراق (Leaf Spot)",
            "ar_desc": "يظهر على شكل بقع دائرية أو غير منتظمة بنية اللون، وغالباً ما يكون ناتجاً عن عدوى فطرية أو بكتيرية تنشط في بيئة عالية الرطوبة.",
            "ar_treatment": "١. تطبيق مبيد فطري واسع المجال.\n٢. توجيه ماء الري للتربة مباشرة وليس للأوراق.\n٣. تنظيف التربة المحيطة من أي أوراق متساقطة.",
            "image": img_spot
        }
    elif "virus" in disease_lower or "mosaic" in disease_lower:
        return {
            "ar_title": "الإصابة الفيروسية (Mosaic Virus)",
            "ar_desc": "الفيروسات تخترق الخلايا وتسبب تجعداً واصفراراً في الأوراق وضعفاً عاماً. لا يوجد علاج كيميائي يقضي على الفيروس بعد الإصابة.",
            "ar_treatment": "١. التدخل الجراحي: اقتلاع النبات المصاب وحرقه فوراً لمنع العدوى.\n٢. مكافحة الحشرات (مثل المن) لأنها الناقل الأساسي للفيروس.\n٣. تعقيم أدوات الزراعة جيداً.",
            "image": img_virus
        }
    elif "mildew" in disease_lower:
        return {
            "ar_title": "البياض الدقيقي/الزغبي (Mildew)",
            "ar_desc": "نمو فطري يظهر كطبقة بيضاء مسحوقية على الأوراق والسيقان، يعيق عملية البناء الضوئي ويضعف الإنتاجية.",
            "ar_treatment": "١. تحسين دورة الهواء بين النباتات.\n٢. رش زيت النيم (Neem Oil) أو مبيد فطري مناسب.\n٣. تقليل الرطوبة وتجنب الري المسائي.",
            "image": img_spot
        }
    else:
        return {
            "ar_title": f"إصابة مُكتشفة: {disease_name}",
            "ar_desc": "النظام استشعر وجود شذوذ بصري في أنسجة النبات يدل على إصابة مرضية تحتاج إلى تدخل سريع لمنع تفاقم الحالة.",
            "ar_treatment": "١. عزل النبات المصاب عن باقي الحقل.\n٢. استشارة خبير زراعي متخصص للتدخل بالمبيد المناسب.\n٣. مراقبة النباتات المجاورة لأي أعراض مشابهة.",
            "image": img_general
        }

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
        rec_details = get_recommendation_details(is_healthy, disease_name)
        
        return jsonify({
            "success": True,
            "raw_class": predicted_class,
            "display_name": display_name,
            "disease_name": disease_name,
            "plant_name": plant_name,
            "confidence": confidence,
            "status": status,
            "is_healthy": is_healthy,
            "ar_title": rec_details["ar_title"],
            "ar_desc": rec_details["ar_desc"],
            "ar_treatment": rec_details["ar_treatment"],
            "rec_image": rec_details["image"]
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
