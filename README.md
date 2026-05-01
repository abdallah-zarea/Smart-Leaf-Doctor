# Smart Leaf Doctor 🌿

**Plant Disease Detection Using Convolutional Neural Networks and Transfer Learning**

مشروع أكاديمي متكامل لاكتشاف أمراض أوراق النباتات (الطماطم والبطاطس) باستخدام الشبكات العصبية العميقة (Deep Learning).

## نظرة عامة على المشروع (Project Overview)
يهدف هذا المشروع إلى توفير أداة ذكية للمساعدة في تشخيص أمراض النباتات. باستخدام تقنيات الذكاء الاصطناعي وتحديداً **الشبكات العصبية التلافيفية (CNN)**، يمكن للنظام التفرقة بين الأوراق السليمة والأوراق المصابة ببعض الأمراض الشائعة بدقة عالية. 

## المشكلة (Problem Statement)
الأمراض النباتية مثل اللفحة المبكرة (Early Blight) واللفحة المتأخرة (Late Blight) تدمر المحاصيل وتسبب خسائر اقتصادية فادحة للمزارعين. التشخيص اليدوي البصري قد يكون بطيئاً ومعرضاً للخطأ. نحتاج إلى نظام ذكي وسريع لتشخيص هذه الأمراض آلياً بمجرد التقاط صورة للورقة.

## مجموعة البيانات (Dataset)
تم الاعتماد على مجموعة بيانات **New Plant Diseases Dataset** المأخوذة من [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset). المشروع الآن يتدرب على جميع الـ **38 فئة (Classes)** المتاحة في مجموعة البيانات والتي تشمل مختلف أمراض النباتات الزراعية (مثل التفاح، الطماطم، البطاطس، العنب، وغيرها).

## مفاهيم الشبكات العصبية المستخدمة (Neural Network Concepts)
- **CNN (Convolutional Neural Network)**: لاستخراج الخصائص (Feature Extraction) من الصور.
- **Transfer Learning (MobileNetV2)**: نقل التعلم باستخدام نموذج قوي مدرب مسبقاً.
- **Data Augmentation**: تكبير وتغيير الصور لتقليل ظاهرة الـ Overfitting.
- **Softmax Function**: لتحويل المخرجات إلى نسب مئوية (احتمالات) لكل فئة.
- **Dropout**: لإيقاف بعض الخلايا العصبية عشوائياً لزيادة قدرة النموذج على التعميم.
- **Optimizer & Loss Function**: استخدام Adam Optimizer و Categorical Crossentropy.

## هيكل المشروع (Folder Structure)
```
Smart_Leaf_Doctor/
│
├── notebooks/
│   └── Smart_Leaf_Doctor_Training.ipynb   # ملف التدريب (Jupyter Notebook)
│
├── app/
│   └── app.py                             # واجهة الويب (Streamlit)
│
├── models/
│   ├── smart_leaf_doctor_mobilenetv2.h5   # النموذج المدرب (يتم إنشاؤه بعد التدريب)
│   └── class_names.json                   # ملف بأسماء الفئات
│
├── outputs/                               # صور النتائج والرسومات البيانية
│
├── dataset/
│   └── README_dataset.md                  # إرشادات تحميل البيانات
│
├── requirements.txt                       # الاعتمادات البرمجية المطلوبة
├── README.md                              # توثيق المشروع
└── project_report.md                      # التقرير الأكاديمي الشامل
```

## خطوات التشغيل والتثبيت (Installation Steps)
1. قم بتثبيت المكتبات المطلوبة:
   ```bash
   pip install -r requirements.txt
   ```
2. تأكد من تحميل مجموعة البيانات باستخدام كاجل أو يدوياً (راجع `dataset/README_dataset.md`).
3. افتح الـ Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/Smart_Leaf_Doctor_Training.ipynb
   ```
4. قم بتشغيل جميع الخلايا في الـ Notebook לتدريب النموذج وحفظه في مجلد `models/`.
5. لتشغيل واجهة الويب (Streamlit App):
   ```bash
   cd app
   streamlit run app.py
   ```

## النتائج (Results)
يحقق نموذج **MobileNetV2** دقة عالية بفضل استخدام Transfer Learning. يمكن الإطلاع على الرسومات البيانية الخاصة بالأداء (Accuracy Curve) والخطأ (Loss Curve) بالإضافة إلى مصفوفة الارتباك (Confusion Matrix) في مجلد `outputs/` بعد إتمام عملية التدريب.

## المحدوديات والأعمال المستقبلية (Limitations & Future Work)
**المحدوديات:**
- عدد الفئات كبير جداً (38 فئة) مما يتطلب وقتاً أطول للتدريب للوصول لدقة عالية جداً.

**العمل المستقبلي:**
- إضافة المزيد من الأمراض وأنواع النباتات.
- استخدام نماذج أثقل مثل ResNet50.
- تطوير تطبيق للهواتف المحمولة (Mobile App) ليستخدمه المزارع في الحقل مباشرة.

---

## 🎓 How to Explain This Project to the Doctor (كيف تشرح المشروع للدكتور)

إليك سكربت بسيط ومباشر للبرزنتيشن:

> "السلام عليكم دكتور. مشروعي بعنوان Smart Leaf Doctor، وهو عبارة عن نظام لاكتشاف أمراض أوراق النباتات (الطماطم والبطاطس) باستخدام الـ CNN.
> 
> استخدمنا الـ CNN لأنها ممتازة جداً في الـ Feature Extraction، يعني تقدر تطلع الخصائص زي بقع المرض وحواف الورقة والألوان لوحدها عن طريق الـ Convolution Layers.
> 
> لتفادي مشكلة الـ Overfitting، استخدمنا الـ Data Augmentation عشان نكبر الداتا عن طريق الـ Rotation والـ Zoom.
> 
> بدلاً من تدريب شبكة من الصفر، استخدمنا **Transfer Learning** وتحديداً نموذج **MobileNetV2**. ده نموذج جاهز ومتدرب على ملايين الصور، فاكتفينا بتعديل الطبقة الأخيرة (Dense layer) واستخدمنا دالة **Softmax** عشان نصنف الصورة لـ 38 فئة مختلفة.
> 
> عملنا تقييم للنموذج ورسمنا الـ Accuracy Curve والـ Loss Curve، وبرضو استخدمنا الـ **Confusion Matrix** عشان نشوف لو الموديل بيتلخبط بين أمراض متشابهة زي الـ Early Blight والـ Late Blight.
> 
> وفي النهاية عملنا Web App بسيط باستخدام Streamlit عشان نخلي الموديل Interactive وتقدر ترفع عليه الصورة ويطلعلك التنبؤ والتوصية العلاجية. شكراً لحضرتك."

---
*Disclaimer: This is an educational academic project built for a university Neural Networks course. It is not a real-world agricultural diagnosis system.*
