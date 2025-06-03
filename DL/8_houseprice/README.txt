مشروع التنبؤ بأسعار المنازل باستخدام الشبكات العصبية
هذا المشروع يستخدم بيانات لخصائص المنازل لتدريب نموذج تنبؤ باستخدام الشبكات العصبية، يهدف إلى تحديد احتمالية ارتفاع سعر المنزل (مخرجات ثنائية: 1 أو 0).

المتطلبات
Python 3.7 أو أحدث

المكتبات المطلوبة:

pandas

numpy

scikit-learn

matplotlib

seaborn

keras

joblib

لتثبيت المتطلبات:


pip install pandas numpy scikit-learn matplotlib seaborn keras joblib
وصف البيانات
البيانات موجودة في ملف housepricedata.csv، وتحتوي على 10 أعمدة من الخصائص مثل عدد الغرف، حجم المنزل، عمره... إلخ، بالإضافة إلى عمود الهدف Y الذي يشير إلى ما إذا كان سعر المنزل مرتفعًا.

خطوات التنفيذ
قراءة البيانات

تقييس الخصائص باستخدام MinMaxScaler

تقسيم البيانات إلى:

70% تدريب

15% تحقق و15% اختبار

بناء ثلاثة نماذج:

نموذج بسيط

نموذج عميق بعدة طبقات

نموذج عميق مع تنظيم L2 وانخفاض عشوائي Dropout

تدريب النماذج ومتابعة الأداء من خلال الدقة والخسارة

رسم نتائج التدريب والتحقق

تقييم النموذج باستخدام Precision وRecall وF1 Score ومصفوفة الالتباس

حفظ النموذج والمقياس

توقع بيانات جديدة



new_house = np.array([[100,6,6,800,2,2,3,7,1,500]])
new_house = min_max_scaler.transform(new_house)
prediction = loaded_model.predict(new_house)
الملفات الناتجة
houses_model.keras — النموذج المدرب

houses_min_max_scaler.pkl — مقياس الخصائص المحفوظ

English
House Price Prediction using Neural Networks
This project uses housing data to train a binary classification neural network that predicts whether the house price is high (1) or not (0).

Requirements
Python 3.7 or later

Required libraries:

pandas

numpy

scikit-learn

matplotlib

seaborn

keras

joblib

To install:

pip install pandas numpy scikit-learn matplotlib seaborn keras joblib
Dataset Description
The dataset is in housepricedata.csv and contains 10 features (e.g., number of rooms, house size, etc.) and one target column Y indicating whether the price is high.

Steps
Load the dataset

Normalize features using MinMaxScaler

Split the data into:

70% training

15% validation and 15% testing

Build three models:

A simple neural network

A deep neural network with multiple layers

A deep network with L2 regularization and Dropout

Train models and track loss and accuracy

Plot training and validation results

Evaluate using Precision, Recall, F1 Score, and Confusion Matrix

Save the model and scaler

Predict on new data

New Prediction Example

new_house = np.array([[100,6,6,800,2,2,3,7,1,500]])
new_house = min_max_scaler.transform(new_house)
prediction = loaded_model.predict(new_house)
Output Files
houses_model.keras — trained model

houses_min_max_scaler.pkl — saved feature scaler

