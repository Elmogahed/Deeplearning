مقدمة
هذا المشروع يهدف إلى بناء نموذج تعلم عميق لتوقع انصراف العملاء من بنك بناءً على بياناتهم. الهدف هو مساعدة البنك في التعرف على العملاء المحتمل أن يغادروا وبالتالي تحسين استراتيجيات الاحتفاظ بالعملاء.

مجموعة البيانات
الملف المستخدم: Churn_Modelling.csv

يحتوي على معلومات العملاء مثل الموقع الجغرافي، الجنس، العمر، مدة الاشتراك، الرصيد، عدد المنتجات، وجود بطاقة ائتمان، النشاط، والراتب المتوقع.

الهدف هو عمود Exited الذي يشير إلى انصراف العميل (0 = لم يغادر، 1 = غادر).

الخصائص المستخدمة
الموقع الجغرافي (Categorical)

الجنس (Categorical)

العمر

مدة الاشتراك

الرصيد

عدد المنتجات

وجود بطاقة ائتمان

هل هو عضو نشط

الراتب المتوقع

خطوات العمل
تحميل البيانات واستكشافها

موازنة البيانات باستخدام RandomOverSampler لتجنب تحيز النموذج نحو الفئة الأكثر انتشارًا.

معالجة البيانات بترميز المتغيرات الفئوية (Label Encoding و One-Hot Encoding) وتوحيد المقاييس.

بناء نموذج الشبكة العصبية باستخدام Keras بطريقتين: نموذج بسيط ونموذج عميق.

تدريب وتقييم النموذج باستخدام مؤشرات مثل الدقة (Accuracy)، الدقة الإيجابية (Precision)، الاستدعاء (Recall)، و F1 Score.

حفظ النموذج والأدوات المستخدمة (مثل المُحوّل والموحد).

التنبؤ ببيانات جديدة باستخدام النموذج المحفوظ.

المتطلبات
Python 3.7 أو أحدث

مكتبات: TensorFlow/Keras، pandas، numpy، matplotlib، scikit-learn، imbalanced-learn، joblib

كيفية التشغيل
تثبيت المكتبات المطلوبة:


pip install tensorflow pandas numpy matplotlib scikit-learn imbalanced-learn joblib
وضع ملف البيانات Churn_Modelling.csv في نفس مجلد السكريبت.

تشغيل السكريبت باستخدام:

python churn_prediction.py
مثال على التنبؤ لعميل جديد

import numpy as np
from joblib import load
from keras.models import load_model

labelencoder_gender = load("churn_label_encoder.pkl")
column_transformer = load("churn_column_transformer.pkl")
scaler = load("churn_standard_scaler.pkl")
model = load_model("churn_model.keras")

new_customer = np.array([[600, "France", "Male", 40, 3, 60000, 2, 1, 1, 50000]], dtype=object)

new_customer[:, 2] = labelencoder_gender.transform(new_customer[:, 2])
new_customer = column_transformer.transform(new_customer)
new_customer = scaler.transform(new_customer)

prob = model.predict(new_customer)
churn = (prob > 0.5)
print("توقع انصراف العميل:", churn)
In English
Overview
This project builds a deep learning model to predict customer churn in a bank based on customer data. The goal is to identify customers likely to leave in order to improve retention strategies.

Dataset
File: Churn_Modelling.csv

Contains customer info such as geography, gender, age, tenure, balance, number of products, credit card ownership, activity, and estimated salary.

Target: Exited column (0 = stayed, 1 = left)

Features Used
Geography (categorical)

Gender (categorical)

Age

Tenure

Balance

Number of Products

Has Credit Card

Is Active Member

Estimated Salary

Steps
Data loading and exploration

Data balancing using RandomOverSampler to avoid model bias towards majority class.

Preprocessing with label encoding and one-hot encoding for categorical variables, and feature scaling.

Building the neural network model using Keras — both a simple and a deeper model.

Training and evaluating with metrics such as accuracy, precision, recall, and F1 score.

Saving the model and preprocessing artifacts (encoder, scaler).

Predicting on new data using the saved model.

Requirements
Python 3.7+

Libraries: TensorFlow/Keras, pandas, numpy, matplotlib, scikit-learn, imbalanced-learn, joblib

How to Run
Install required packages:

pip install tensorflow pandas numpy matplotlib scikit-learn imbalanced-learn joblib
Place the Churn_Modelling.csv file in the script directory.

Run the script:

python churn_prediction.py
Example Usage for New Customer Prediction

import numpy as np
from joblib import load
from keras.models import load_model

labelencoder_gender = load("churn_label_encoder.pkl")
column_transformer = load("churn_column_transformer.pkl")
scaler = load("churn_standard_scaler.pkl")
model = load_model("churn_model.keras")

new_customer = np.array([[600, "France", "Male", 40, 3, 60000, 2, 1, 1, 50000]], dtype=object)

new_customer[:, 2] = labelencoder_gender.transform(new_customer[:, 2])
new_customer = column_transformer.transform(new_customer)
new_customer = scaler.transform(new_customer)

prob = model.predict(new_customer)
churn = (prob > 0.5)
print("Churn prediction:", churn)