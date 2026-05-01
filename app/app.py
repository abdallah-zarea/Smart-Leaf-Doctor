import streamlit as st
import numpy as np
from PIL import Image
import json
import os
import tensorflow as tf
import base64

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Smart Leaf Doctor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. AGGRESSIVE CSS INJECTION FOR PREMIUM UI ---
# Color Palette: Deep Blue #0A0F1C, Neon Green #00FFA3, Cyan #00C9FF
custom_css = """
<style>
    /* Global Font & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    .stApp {
        background-color: #0A0F1C !important;
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(0, 255, 163, 0.05), transparent 25%),
            radial-gradient(circle at 85% 30%, rgba(0, 201, 255, 0.05), transparent 25%);
        color: #FFFFFF !important;
    }

    /* Hide standard Streamlit elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 15, 28, 0.95) !important;
        border-right: 1px solid rgba(0, 255, 163, 0.2);
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 255, 163, 0.1);
        border: 1px solid rgba(0, 255, 163, 0.2);
    }
    
    /* Neon Glow Text */
    .neon-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00FFA3, #00C9FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 0px;
        line-height: 1.1;
    }
    .neon-subtitle {
        color: #8892B0;
        font-size: 1.2rem;
        font-weight: 300;
        margin-top: 5px;
        margin-bottom: 40px;
        letter-spacing: 1px;
    }

    /* Style the File Uploader */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(0, 255, 163, 0.02) !important;
        border: 2px dashed rgba(0, 255, 163, 0.3) !important;
        border-radius: 12px;
        padding: 40px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: rgba(0, 255, 163, 0.05) !important;
        border: 2px dashed #00FFA3 !important;
        box-shadow: 0 0 20px rgba(0, 255, 163, 0.2);
    }
    
    /* Primary Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, rgba(0, 255, 163, 0.1), rgba(0, 201, 255, 0.1));
        border: 1px solid #00FFA3 !important;
        color: #00FFA3 !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 12px 24px;
        width: 100%;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #00FFA3, #00C9FF);
        color: #0A0F1C !important;
        box-shadow: 0 0 15px rgba(0, 255, 163, 0.5);
        transform: scale(1.02);
    }

    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #00FFA3 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #8892B0 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Status Badges */
    .status-healthy {
        background: rgba(0, 255, 163, 0.1);
        border: 1px solid #00FFA3;
        color: #00FFA3;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 0 15px rgba(0, 255, 163, 0.2);
    }
    .status-diseased {
        background: rgba(255, 51, 102, 0.1);
        border: 1px solid #FF3366;
        color: #FF3366;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 0 15px rgba(255, 51, 102, 0.2);
    }
    
    /* Progress Bar */
    .progress-bg {
        width: 100%;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        margin-top: 10px;
    }
    .progress-fill {
        height: 6px;
        background: linear-gradient(90deg, #00C9FF, #00FFA3);
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 255, 163, 0.5);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
@st.cache_resource
def load_model():
    model_path = '../models/smart_leaf_doctor_mobilenetv2.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

@st.cache_data
def load_class_names():
    json_path = '../models/class_names.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return []

model = load_model()
class_names = load_class_names()

def get_recommendation(is_healthy, disease_name):
    if is_healthy:
        return "Plant vitality optimal. Tissue structure is robust. Maintain current hydration and light cycles."
    
    d_low = disease_name.lower()
    if "blight" in d_low: return "Fungal pathogen detected. Isolate plant. Apply copper-based fungicidal spray immediately."
    if "rust" in d_low: return "Rust spores identified. Remove necrotic tissue. Apply sulfur fungicide."
    if "virus" in d_low or "mosaic" in d_low: return "Viral signature confirmed. Uncurable. Eradicate specimen to protect surrounding crop."
    return "Anomaly detected. Initiate quarantine protocols and apply broad-spectrum treatment."

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="neon-title" style="font-size: 1.5rem;">SLD_CORE</div>', unsafe_allow_html=True)
    st.markdown('<div style="color: #00FFA3; font-family: monospace; font-size: 0.8rem; margin-bottom: 30px;">SYSTEM STATUS: ONLINE</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Project Overview")
    st.markdown("<span style='color: #8892B0; font-size:0.9rem;'>AI-driven agricultural diagnostic tool. Scans biological tissue for anomalies across 38 pathogen classes.</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Neural Workflow")
    st.markdown("<span style='color: #8892B0; font-size:0.9rem;'>1. Image Ingestion<br>2. CNN Feature Extraction<br>3. Softmax Classification</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Model Info")
    st.markdown("<span style='color: #8892B0; font-size:0.9rem;'>Architecture: MobileNetV2<br>Weights: ImageNet<br>Parameters: ~2.5M</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 5. HERO SECTION ---
st.markdown('<div class="neon-title">Smart Leaf Doctor</div>', unsafe_allow_html=True)
st.markdown('<div class="neon-subtitle">AI-POWERED DIAGNOSTIC ENGINE v2.4</div>', unsafe_allow_html=True)

# --- 6. MAIN WORK AREA (Split Layout) ---
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="glass-card" style="min-height: 450px;">', unsafe_allow_html=True)
    st.markdown("<h3 style='color: #FFFFFF; font-weight: 600;'>📤 INPUT DATASET</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop leaf image here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True, clamp=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card" style="min-height: 450px;">', unsafe_allow_html=True)
    st.markdown("<h3 style='color: #00FFA3; font-weight: 600;'>🧠 AI ANALYSIS PANEL</h3>", unsafe_allow_html=True)
    
    if not uploaded_file:
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height: 300px; color: #8892B0; font-family: monospace;">
            <div style="font-size: 3rem; margin-bottom: 20px;">⏱️</div>
            <div>AWAITING VISUAL INPUT...</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button("INITIALIZE NEURAL SCAN"):
            if model is None:
                st.error("CORE MODULE MISSING: AI Model not loaded.")
            else:
                with st.spinner("Executing convolutional layers..."):
                    try:
                        # Preprocessing
                        img_resized = image.resize((224, 224))
                        if img_resized.mode != 'RGB':
                            img_resized = img_resized.convert('RGB')
                        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                        img_array = np.expand_dims(img_array, axis=0) / 255.0
                        
                        # Prediction
                        predictions = model.predict(img_array)[0]
                        pred_idx = np.argmax(predictions)
                        confidence = float(predictions[pred_idx] * 100)
                        
                        raw_class = class_names[pred_idx] if len(class_names) > 0 else f"Class_{pred_idx}"
                        parts = raw_class.split('___')
                        plant = parts[0].replace('_', ' ')
                        disease = parts[-1].replace('_', ' ') if len(parts) > 1 else plant
                        
                        is_healthy = "healthy" in raw_class.lower()
                        
                        # --- RESULT DISPLAY ---
                        st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
                        
                        # Status Badge
                        status_html = '<div class="status-healthy">HEALTHY</div>' if is_healthy else '<div class="status-diseased">DISEASED</div>'
                        st.markdown(status_html, unsafe_allow_html=True)
                        
                        st.markdown(f"<h2 style='margin-top: 15px; margin-bottom: 0;'>{disease}</h2>", unsafe_allow_html=True)
                        st.markdown(f"<div style='color: #8892B0; font-family: monospace;'>PLANT SPECIMEN: {plant.upper()}</div>", unsafe_allow_html=True)
                        
                        # Confidence Bar
                        st.markdown(f"""
                        <div style="margin-top: 25px;">
                            <div style="display: flex; justify-content: space-between; font-family: monospace;">
                                <span style="color: #8892B0;">CONFIDENCE MATRIX</span>
                                <span style="color: #00FFA3; font-weight: bold;">{confidence:.1f}%</span>
                            </div>
                            <div class="progress-bg">
                                <div class="progress-fill" style="width: {confidence}%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # AI Recommendation
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<div class="glass-card" style="background: rgba(0, 201, 255, 0.05); border-left: 4px solid #00C9FF; padding: 15px;">', unsafe_allow_html=True)
                        st.markdown("<div style='color: #00C9FF; font-weight: bold; font-size: 0.8rem; margin-bottom: 5px; letter-spacing: 1px;'>💡 AI PRESCRIPTION</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size: 0.95rem; line-height: 1.5;'>{get_recommendation(is_healthy, disease)}</div>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"SYSTEM FAILURE: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 7. ANALYTICS SECTION ---
st.markdown("<br><h3 style='text-align:center; color:#8892B0; letter-spacing: 2px; font-weight: 300; font-size: 1rem;'>NEURAL NETWORK ANALYTICS</h3>", unsafe_allow_html=True)

a1, a2 = st.columns(2)
with a1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color: #00FFA3;'>Model Accuracy</h4>", unsafe_allow_html=True)
    if os.path.exists('../outputs/accuracy_curve.png'):
        st.image('../outputs/accuracy_curve.png', use_column_width=True)
    else:
        st.markdown("<div style='color: #8892B0; text-align: center; padding: 40px;'>Analytics pending training completion...</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with a2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color: #00C9FF;'>Loss Optimization</h4>", unsafe_allow_html=True)
    if os.path.exists('../outputs/loss_curve.png'):
        st.image('../outputs/loss_curve.png', use_column_width=True)
    else:
        st.markdown("<div style='color: #8892B0; text-align: center; padding: 40px;'>Analytics pending training completion...</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
