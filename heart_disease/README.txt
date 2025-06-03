عنوان المشروع: التنبؤ بأمراض القلب باستخدام الشبكات العصبية
نظرة عامة:
يهدف هذا المشروع إلى بناء نموذج تنبؤي لاكتشاف وجود مرض في القلب اعتمادًا على بيانات طبية من مجموعة بيانات UCI. يشمل المشروع مراحل المعالجة المسبقة للبيانات، تحليلها، تدريب النموذج، حفظ النموذج والأدوات، واختبار التنبؤ على بيانات جديدة.
1. تحميل البيانات وفحصها
استيراد مكتبات pandas، matplotlib، seaborn، وnumpy.

قراءة ملف البيانات heart_disease_uci.csv باستخدام pandas.read_csv.

عرض أول 5 صفوف من البيانات لمعرفة شكلها.

طباعة معلومات عن الأعمدة وأنواع البيانات وعدد القيم المفقودة.

التحقق من وجود صفوف مكررة أو قيم مفقودة.

2. استكشاف البيانات بصريًا
رسم توزيعات (Histograms) لأعمدة ضغط الدم (trestbps)، الكوليسترول (chol)، معدل ضربات القلب (thalch) ومؤشر oldpeak.

استخدام مخططات صندوق (Boxplots) لتحديد القيم الشاذة (Outliers) في نفس الأعمدة.

3. معالجة القيم المفقودة (Null Values)
ملء القيم الفارغة في الأعمدة الرقمية بالوسيط (Median).

ملء القيم الفارغة في الأعمدة الفئوية بأكثر قيمة تكرارًا (Mode).

إعادة فحص البيانات للتأكد من إزالة القيم الفارغة.

4. موازنة بيانات التصنيف (Data Balancing)
فحص توزيع الفئات في عمود الهدف num (وجود مرض أو لا).

تحويل القيم إلى ثنائية: 0 (لا مرض) و1 (مرض).

استخدام تقنية إعادة العينة العشوائية (RandomOverSampler) لموازنة الفئات غير المتساوية.

5. تجهيز البيانات للنموذج (Feature Engineering)
فصل البيانات إلى مدخلات X وهدف y.

تحويل الأعمدة النصية إلى أرقام باستخدام LabelEncoder وOneHotEncoder (مثلاً عمود الجنس sex وتحويل الأعمدة المتعددة الفئات).

تقسيم البيانات إلى مجموعات تدريب، تحقق، واختبار (70% تدريب، 15% تحقق، 15% اختبار).

توحيد مقياس البيانات (Standardization) باستخدام StandardScaler.

6. بناء نموذج الشبكة العصبية (Neural Network)
إنشاء نموذج Sequential باستخدام Keras.

إضافة طبقات كثيفة (Dense) مع دوال تفعيل relu وطبقات Dropout لتجنب الإفراط في التعلم.

استخدام تنظيم L2 كنوع من الـ regularization.

طبقة إخراج واحدة مع دالة تفعيل sigmoid للتصنيف الثنائي.

تجميع النموذج باستخدام خوارزمية التحسين sgd وخسارة binary_crossentropy.

7. تدريب وتقييم النموذج
تدريب النموذج على بيانات التدريب مع التحقق على بيانات التحقق.

رسم منحنيات الدقة والخسارة لكل من التدريب والتحقق عبر epochs.

تقييم النموذج على مجموعة الاختبار.

حساب دقة النموذج، الدقة الإيجابية، الاستدعاء، وF1-score.

عرض مصفوفة الالتباس (Confusion Matrix).

8. حفظ النموذج والأدوات
حفظ النموذج المدرب باستخدام .save().

حفظ أدوات الترميز (LabelEncoder)، التحويلات (ColumnTransformer)، ومُوحد القياس (StandardScaler) باستخدام joblib.

9. اختبار النموذج على بيانات جديدة
تحميل الأدوات المحفوظة.

تجهيز بيانات المريض الجديد بنفس خطوات التحويل والتوحيد.

استخدام النموذج المحفوظ للتنبؤ بوجود مرض القلب للمريض الجديد.

 Networks
1. Data Loading and Inspection
Import necessary libraries: pandas, matplotlib, seaborn, and numpy.

Read the dataset heart_disease_uci.csv.

Display the first 5 rows to get an overview of the data.

Print data information: column types, number of non-null values.

Check for duplicated rows and missing values.

2. Exploratory Data Analysis (EDA)
Plot histograms for numerical features: trestbps, chol, thalch, and oldpeak to visualize their distributions.

Use boxplots to identify outliers in the same columns.

3. Handling Missing Values
Fill missing numeric values using the median.

Fill missing categorical values using the mode (most frequent value).

Re-check to confirm that no missing values remain.

4. Balancing the Dataset
Check class distribution in the target column num (presence of heart disease).

Convert the target to binary: 0 for no disease, 1 for disease.

Use RandomOverSampler from imblearn to balance the class distribution.

5. Feature Engineering and Preprocessing
Split the data into X (features) and y (target).

Use LabelEncoder to convert categorical sex values to numerical.

Convert some boolean features to integers (fbs, exang).

Apply OneHotEncoder for multi-class categorical columns like cp, thal, etc., using ColumnTransformer.

Split the dataset into train, validation, and test sets (70/15/15 split).

Standardize features using StandardScaler.

6. Building the Neural Network Model
Create a Sequential model using Keras.

Add:

Dense layers with ReLU activation.

Dropout layers to prevent overfitting.

L2 regularization for stability.

Final output layer with sigmoid activation for binary classification.

Compile using SGD optimizer and binary crossentropy loss.

7. Training and Evaluation
Train the model on the training data and validate on the validation set.

Plot training and validation accuracy and loss over epochs.

Evaluate the model on the test set.

Calculate:

Accuracy

Precision

Recall

F1 Score

Plot the Confusion Matrix using seaborn.

8. Saving the Model and Encoders
Save the trained model (heart_disease.keras).

Save the LabelEncoder, ColumnTransformer, and StandardScaler using joblib.

9. Making Predictions on New Patient Data
Load the saved model and preprocessing tools.

Input new patient data and apply the same transformations.

Predict heart disease probability and return binary result.

