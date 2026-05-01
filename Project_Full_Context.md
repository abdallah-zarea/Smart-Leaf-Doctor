# Smart Leaf Doctor - Full Project Context

هذا الملف مخصص لنسخه وإعطائه لأي نظام ذكاء اصطناعي (مثل ChatGPT أو Claude) لشرح المشروع، وهو يحتوي على كل تفاصيل الأكواد والأدوات المستخدمة.

## الفكرة العامة
مشروع يهدف إلى تصنيف 38 فئة مختلفة من أوراق النباتات (مريضة وسليمة) باستخدام الذكاء الاصطناعي وتحديداً الشبكات العصبية (CNN) ونقل التعلم (Transfer Learning) عبر نموذج MobileNetV2.

---

## 1. كود التدريب (الـ Notebook الأساسي)
الكود أدناه هو المحتوى الفعلي لملف Jupyter Notebook، قمت بتحويله إلى كود Python لتسهيل قراءته:

```python
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import json

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

# --- 1. Dataset Loading ---
LOCAL_DATASET_DIR = '../dataset/'
BASE_DIR = None

try:
    import kagglehub
    path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
    BASE_DIR = os.path.join(path, "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)")
except Exception as e:
    BASE_DIR = os.path.join(LOCAL_DATASET_DIR, 'New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)')

train_dir = os.path.join(BASE_DIR, 'train')
valid_dir = os.path.join(BASE_DIR, 'valid')

# --- 2. Data Augmentation ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(
    rescale=1./255,          
    rotation_range=20,       
    zoom_range=0.2,          
    horizontal_flip=True,    
    brightness_range=[0.8, 1.2]
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)

os.makedirs('../models', exist_ok=True)
with open('../models/class_names.json', 'w') as f:
    json.dump(class_names, f)

# --- 3. MobileNetV2 Transfer Learning ---
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

mobilenet_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

mobilenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('../models/smart_leaf_doctor_mobilenetv2.h5', monitor='val_accuracy', save_best_only=True)
]

history_mobilenet = mobilenet_model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10,
    callbacks=callbacks
)
```

---

## 2. كود واجهة الويب (Streamlit App)
الكود المتواجد في `app/app.py`:

```python
import streamlit as st
import numpy as np
from PIL import Image
import json
import os
import tensorflow as tf

st.set_page_config(page_title="Smart Leaf Doctor", page_icon="🌿", layout="wide")

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

st.title("🌿 Smart Leaf Doctor")
st.subheader("Plant Disease Detection Using Neural Networks")

st.sidebar.title("👨‍🔬 About the Project")
st.sidebar.info(
    "This is a deep learning web application designed to identify plant leaf diseases. "
    "It uses a Convolutional Neural Network (MobileNetV2) trained on all 38 specific classes "
    "of various plant leaves."
)

st.markdown("### Upload a Leaf Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        if st.button("Predict 🚀", type="primary"):
            if model is None:
                st.error("Model file not found.")
            else:
                with st.spinner("Analyzing image using Neural Network..."):
                    img_resized = image.resize((224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0
                    
                    predictions = model.predict(img_array)[0]
                    pred_idx = np.argmax(predictions)
                    predicted_class = class_names[pred_idx]
                    confidence = predictions[pred_idx] * 100
                    
                    status = "Healthy 🌱" if "healthy" in predicted_class.lower() else "Diseased 🦠"
                    
                    st.success("Analysis Complete!")
                    st.metric(label="Predicted Class", value=predicted_class)
                    st.metric(label="Confidence", value=f"{confidence:.2f}%")
                    st.metric(label="Status", value=status)
```

---
## 3. ملف requirements.txt
```txt
tensorflow
streamlit
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
kaggle
opencv-python-headless==4.9.0.80
notebook
```
