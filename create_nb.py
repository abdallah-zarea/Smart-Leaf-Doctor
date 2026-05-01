import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# مشروع Smart Leaf Doctor: اكتشاف أمراض أوراق النباتات 🌿\n",
    "\n",
    "**الهدف من المشروع:**\n",
    "بناء نظام ذكي يعتمد على الشبكات العصبية التلافيفية (CNN) ونقل التعلم (Transfer Learning) لاكتشاف جميع الأمراض والفئات (38 فئة) المتوفرة في مجموعة البيانات.\n",
    "\n",
    "**لماذا نستخدم نقل التعلم (Transfer Learning)؟**\n",
    "نقل التعلم يعني استخدام نموذج مدرب مسبقاً (MobileNetV2). هذا النموذج قوي جداً في استخراج الخصائص مما يمكننا من تصنيف عدد كبير من الفئات (38 فئة) بدقة وكفاءة عالية."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import random\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import cv2\n",
    "\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.models import Sequential, Model\n",
    "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D\n",
    "from tensorflow.keras.applications import MobileNetV2\n",
    "from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "np.random.seed(42)\n",
    "tf.random.set_seed(42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. تحميل وتجهيز مجموعة البيانات (Dataset Loading)\n",
    "\n",
    "سنحاول أولاً استخدام `kaggle` لتحميل البيانات، وفي حالة نجاحه أو وجود البيانات محلياً، سنستخدم جميع الفئات (المجلدات) المتوفرة للتدريب والاختبار."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "LOCAL_DATASET_DIR = '../dataset/'\n",
    "BASE_DIR = None\n",
    "\n",
    "try:\n",
    "    import kagglehub\n",
    "    if not os.path.exists(os.path.join(LOCAL_DATASET_DIR, 'New Plant Diseases Dataset(Augmented)')):\n",
    "        print(\"جاري تحميل مجموعة البيانات من كاجل...\")\n",
    "        path = kagglehub.dataset_download(\"vipoooool/new-plant-diseases-dataset\")\n",
    "        print(\"تم التحميل بنجاح في:\", path)\n",
    "        BASE_DIR = os.path.join(path, \"New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)\")\n",
    "    else:\n",
    "        BASE_DIR = os.path.join(LOCAL_DATASET_DIR, 'New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)')\n",
    "        print(\"البيانات موجودة محلياً في:\", BASE_DIR)\n",
    "except Exception as e:\n",
    "    print(f\"حدث خطأ أثناء محاولة تحميل البيانات: {e}\")\n",
    "    print(\"الرجاء وضع البيانات يدوياً في مجلد dataset/\")\n",
    "    BASE_DIR = os.path.join(LOCAL_DATASET_DIR, 'New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_dir = None
valid_dir = None
if BASE_DIR and os.path.exists(BASE_DIR):\n",
    "    train_dir = os.path.join(BASE_DIR, 'train')\n",
    "    valid_dir = os.path.join(BASE_DIR, 'valid')\n",
    "    \n",
    "    print(\"تم إعداد مسارات التدريب والاختبار لجميع الفئات.\")\n",
    "    \n",
    "    # تأكيد عدد الصور\n",
    "    for split_dir, split_name in [(train_dir, \"Training\"), (valid_dir, \"Validation\")]:\n",
    "        print(f\"\\n--- {split_name} Set ---\")\n",
    "        total = 0\n",
    "        if os.path.exists(split_dir):\n",
    "            classes = os.listdir(split_dir)\n",
    "            print(f\"Number of classes: {len(classes)}\")\n",
    "            for c in classes:\n",
    "                class_path = os.path.join(split_dir, c)\n",
    "                if os.path.isdir(class_path):\n",
    "                    count = len(os.listdir(class_path))\n",
    "                    total += count\n",
    "            print(f\"Total: {total} images\")\n",
    "else:\n",
    "    print(\"الرجاء التأكد من مسار البيانات. قد تحتاج إلى تحميل البيانات يدوياً.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. المعالجة المسبقة وتكبير البيانات (Preprocessing & Data Augmentation)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "IMAGE_SIZE = (224, 224)\n",
    "BATCH_SIZE = 64 # تم زيادة حجم الدفعة نظراً لكبر حجم البيانات لتسريع التدريب\n",
    "\n",
    "if train_dir and os.path.exists(train_dir):\n",
    "    train_datagen = ImageDataGenerator(\n",
    "        rescale=1./255,          \n",
    "        rotation_range=20,       \n",
    "        zoom_range=0.2,          \n",
    "        horizontal_flip=True,    \n",
    "        brightness_range=[0.8, 1.2]\n",
    "    )\n",
    "    \n",
    "    valid_datagen = ImageDataGenerator(rescale=1./255)\n",
    "    \n",
    "    train_generator = train_datagen.flow_from_directory(\n",
    "        train_dir,\n",
    "        target_size=IMAGE_SIZE,\n",
    "        batch_size=BATCH_SIZE,\n",
    "        class_mode='categorical'\n",
    "    )\n",
    "    \n",
    "    valid_generator = valid_datagen.flow_from_directory(\n",
    "        valid_dir,\n",
    "        target_size=IMAGE_SIZE,\n",
    "        batch_size=BATCH_SIZE,\n",
    "        class_mode='categorical',\n",
    "        shuffle=False\n",
    "    )\n",
    "    \n",
    "    class_names = list(train_generator.class_indices.keys())\n",
    "    num_classes = len(class_names)\n",
    "    print(f\"تم العثور على {num_classes} فئة.\")\n",
    "    \n",
    "    import json\n",
    "    os.makedirs('../models', exist_ok=True)\n",
    "    with open('../models/class_names.json', 'w') as f:\n",
    "        json.dump(class_names, f)\n",
    "    print(\"تم حفظ أسماء الفئات في models/class_names.json\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. نموذج MobileNetV2 (Transfer Learning)\n",
    "\n",
    "نظراً لضخامة البيانات (38 فئة)، سنتخطى النموذج البسيط ونستخدم MobileNetV2 مباشرة للحصول على أفضل دقة وتوفير الوقت."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if train_dir and os.path.exists(train_dir) and 'num_classes' in locals():\n",
    "    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))\n",
    "    base_model.trainable = False # تجميد الأوزان الأساسية\n",
    "    \n",
    "    mobilenet_model = Sequential([\n",
    "        base_model,\n",
    "        GlobalAveragePooling2D(),\n",
    "        Dense(512, activation='relu'),\n",
    "        Dropout(0.4),\n",
    "        Dense(num_classes, activation='softmax') # عدد ديناميكي يمثل جميع الفئات\n",
    "    ])\n",
    "    \n",
    "    mobilenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n",
    "    mobilenet_model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if os.path.exists(train_dir) and 'mobilenet_model' in locals():\n",
    "    callbacks = [\n",
    "        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),\n",
    "        ModelCheckpoint('../models/smart_leaf_doctor_mobilenetv2.h5', monitor='val_accuracy', save_best_only=True)\n",
    "    ]\n",
    "    \n",
    "    epochs = 10 \n",
    "    history_mobilenet = mobilenet_model.fit(\n",
    "        train_generator,\n",
    "        validation_data=valid_generator,\n",
    "        epochs=epochs,\n",
    "        callbacks=callbacks\n",
    "    )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. تقييم النموذج (Model Evaluation)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if os.path.exists(train_dir) and 'history_mobilenet' in locals():\n",
    "    os.makedirs('../outputs', exist_ok=True)\n",
    "    \n",
    "    # منحنى الدقة (Accuracy Curve)\n",
    "    plt.figure(figsize=(8, 6))\n",
    "    plt.plot(history_mobilenet.history['accuracy'], label='Training Accuracy')\n",
    "    plt.plot(history_mobilenet.history['val_accuracy'], label='Validation Accuracy')\n",
    "    plt.title('منحنى الدقة (Model Accuracy)')\n",
    "    plt.xlabel('Epoch')\n",
    "    plt.ylabel('Accuracy')\n",
    "    plt.legend()\n",
    "    plt.savefig('../outputs/accuracy_curve.png')\n",
    "    plt.show()\n",
    "    \n",
    "    # منحنى الخطأ (Loss Curve)\n",
    "    plt.figure(figsize=(8, 6))\n",
    "    plt.plot(history_mobilenet.history['loss'], label='Training Loss')\n",
    "    plt.plot(history_mobilenet.history['val_loss'], label='Validation Loss')\n",
    "    plt.title('منحنى الخطأ (Model Loss)')\n",
    "    plt.xlabel('Epoch')\n",
    "    plt.ylabel('Loss')\n",
    "    plt.legend()\n",
    "    plt.savefig('../outputs/loss_curve.png')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if os.path.exists(train_dir) and 'history_mobilenet' in locals():\n",
    "    # الحصول على التنبؤات\n",
    "    valid_generator.reset()\n",
    "    predictions = mobilenet_model.predict(valid_generator)\n",
    "    y_pred = np.argmax(predictions, axis=1)\n",
    "    y_true = valid_generator.classes\n",
    "    \n",
    "    # مصفوفة الارتباك (سيتم حفظها كصورة نظراً لضخامتها)\n",
    "    cm = confusion_matrix(y_true, y_pred)\n",
    "    plt.figure(figsize=(20, 18))\n",
    "    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')\n",
    "    plt.title('مصفوفة الارتباك (Confusion Matrix)')\n",
    "    plt.xlabel('Predicted')\n",
    "    plt.ylabel('True')\n",
    "    plt.savefig('../outputs/confusion_matrix.png')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. دالة التنبؤ وتجربتها"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.preprocessing import image\n",
    "\n",
    "if os.path.exists(train_dir) and 'mobilenet_model' in locals():\n",
    "    def predict_leaf_disease(img_path):\n",
    "        img = image.load_img(img_path, target_size=(224, 224))\n",
    "        img_array = image.img_to_array(img)\n",
    "        img_array = np.expand_dims(img_array, axis=0) / 255.0\n",
    "        \n",
    "        preds = mobilenet_model.predict(img_array)[0]\n",
    "        pred_idx = np.argmax(preds)\n",
    "        predicted_class = class_names[pred_idx]\n",
    "        confidence = preds[pred_idx] * 100\n",
    "        \n",
    "        status = \"Healthy\" if \"healthy\" in predicted_class.lower() else \"Diseased\"\n",
    "        return predicted_class, confidence, status\n",
    "\n",
    "    # عرض عينة (Sample Prediction)\n",
    "    sample_class = random.choice(class_names)\n",
    "    sample_dir = os.path.join(valid_dir, sample_class)\n",
    "    if os.path.exists(sample_dir) and len(os.listdir(sample_dir)) > 0:\n",
    "        sample_img = random.choice(os.listdir(sample_dir))\n",
    "        sample_path = os.path.join(sample_dir, sample_img)\n",
    "        \n",
    "        predicted_class, confidence, status = predict_leaf_disease(sample_path)\n",
    "        \n",
    "        img = cv2.imread(sample_path)\n",
    "        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "        plt.imshow(img)\n",
    "        plt.title(f\"True: {sample_class}\\nPred: {predicted_class}\\nConf: {confidence:.1f}%\")\n",
    "        plt.axis('off')\n",
    "        plt.savefig('../outputs/sample_predictions.png')\n",
    "        plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('/Users/abdallahzarea/Desktop/NN Project/Smart_Leaf_Doctor/notebooks/Smart_Leaf_Doctor_Training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
