عنوان المشروع: التنبؤ بأمراض القلب باستخدام الشبكات العصبية
نظرة عامة:
يهدف هذا المشروع إلى بناء نموذج تنبؤي لاكتشاف وجود مرض في القلب اعتمادًا على بيانات طبية من مجموعة بيانات UCI. يشمل المشروع مراحل المعالجة المسبقة للبيانات، تحليلها، تدريب النموذج، حفظ النموذج والأدوات، واختبار التنبؤ على بيانات جديدة.

مكونات المشروع:
تنظيف البيانات والتعامل مع القيم المفقودة.

تحليل بصري للبيانات (مخططات التوزيع والصناديق).

موازنة البيانات لتقليل التحيّز.

بناء نموذج شبكة عصبية باستخدام Keras.

تقييم أداء النموذج.

حفظ النموذج وملفاته (مثل أدوات التحويل والتقييس).

اختبار النموذج على مريض جديد.

الأدوات المستخدمة:
Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Imbalanced-learn

TensorFlow / Keras

Joblib

خطوات التشغيل:
تثبيت المكتبات المطلوبة:


pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow joblib
تشغيل ملف المشروع الرئيسي:


python heart_disease_prediction.py
يمكن استخدام النموذج المحفوظ للتنبؤ بحالة مريض جديد بعد تحويل البيانات بنفس الأدوات المستخدمة أثناء التدريب.

English Version (No symbols or emojis):
Project Title: Heart Disease Prediction Using Neural Networks
Overview:
This project builds a machine learning model to predict heart disease presence based on clinical data from the UCI dataset. The workflow includes preprocessing, visualization, model training, artifact saving, and prediction on new data.

Project Components:
Handling missing values and data cleaning

Visualizing distributions and outliers

Addressing class imbalance using resampling

Building a neural network model using Keras

Model evaluation and metrics

Saving model and preprocessing tools

Performing inference on new patient data

Tools and Libraries:
Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Imbalanced-learn

TensorFlow / Keras

Joblib

How to Run:
Install the required packages:


pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow joblib
Run the main script:

python heart_disease_prediction.py
Use the saved model and preprocessors to make predictions on new input data.

