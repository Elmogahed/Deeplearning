مشروع التنبؤ باستجابة العملاء لحملة تسويقية باستخدام الشبكات العصبية
هذا المشروع يستخدم مجموعة بيانات لعملاء شركة تسويق من أجل تدريب نموذج تصنيفي يحدد احتمالية استجابة العميل (نعم أو لا) لحملة تسويقية.

المتطلبات
Python 3.7 أو أحدث

المكتبات المطلوبة:

numpy

pandas

scikit-learn

matplotlib

keras

imbalanced-learn

لتثبيت المتطلبات:

bash
نسخ
تحرير
pip install numpy pandas scikit-learn matplotlib keras imbalanced-learn
وصف البيانات
البيانات موجودة في ملف customers.csv، وتحتوي على خصائص ديموغرافية وسلوكية للعملاء، والعمود المستهدف y يشير إلى ما إذا كان العميل قد استجاب للحملة (yes أو no).

خطوات التنفيذ
قراءة البيانات واستكشافها

التعامل مع التوزيع غير المتوازن باستخدام RandomOverSampler

تحويل الخصائص الفئوية باستخدام OneHotEncoding بالتكرار لجميع الأعمدة الفئوية

توحيد الخصائص الرقمية باستخدام StandardScaler

تقسيم البيانات إلى مجموعة تدريب واختبار

بناء نموذج شبكة عصبية متعددة الطبقات

طبقات خفية بعدد وحدات 128, 64, 32

الطبقة الأخيرة تستخدم دالة تنشيط sigmoid للتصنيف الثنائي

استخدام طبقة Dropout لتقليل التخصيص الزائد

تدريب النموذج باستخدام binary_crossentropy و adam كخوارزمية تحسين

عرض نتائج التدريب باستخدام الرسوم البيانية للدقة

تقييم النموذج باستخدام المقاييس: الدقة، الاستدعاء، معامل الدقة، وF1 Score

المخرجات
عرض للدقة أثناء التدريب والتحقق

تقييم النموذج على بيانات الاختبار

طباعة التنبؤات وتحويلها إلى مخرجات ثنائية

حساب المقاييس الأساسية لأداء النموذج

مثال للتنبؤ
python


y_pred = classifier.predict(X_test)
y_pred_binary = (y_pred > 0.5)
English
Customer Response Prediction using Neural Networks
This project builds a binary classification model using customer data to predict whether a client will respond to a marketing campaign.

Requirements
Python 3.7 or later

Required libraries:

numpy

pandas

scikit-learn

matplotlib

keras

imbalanced-learn

To install:


pip install numpy pandas scikit-learn matplotlib keras imbalanced-learn
Dataset Description
The dataset (customers.csv) includes demographic and behavioral features of customers. The target column y indicates whether the customer responded (yes or no).

Workflow
Read and inspect the dataset

Handle class imbalance using RandomOverSampler

One-hot encode categorical features with multiple passes

Standardize numerical features using StandardScaler

Split dataset into training and test sets

Build neural network model

Hidden layers: 128, 64, 32 neurons

Output layer: sigmoid activation

Dropout layer to prevent overfitting

Train the model using binary_crossentropy and adam optimizer

Visualize accuracy during training

Evaluate model with accuracy, precision, recall, and F1 Score

Outputs
Accuracy and validation accuracy plots

Test set evaluation metrics

Binary predictions from model

Full performance report

Example Prediction

y_pred = classifier.predict(X_test)
y_pred_binary = (y_pred > 0.5)