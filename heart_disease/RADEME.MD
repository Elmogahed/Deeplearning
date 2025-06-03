#  مشروع التنبؤ بأمراض القلب باستخدام الشبكات العصبية

##  المتطلبات

```bash
pip install pandas numpy matplotlib seaborn scikit-learn keras tensorflow imbalanced-learn joblib
```

---

##  قراءة البيانات واستكشافها

نقوم بقراءة الملف، عرض أول خمس صفوف، معرفة أنواع البيانات، وعدد القيم المكررة أو المفقودة.

```python
df = pd.read_csv('heart_disease_uci.csv')
print(df.head())
print(df.info())
print(df.duplicated().sum())
print(df.isnull().sum())
```

---

##  التحليل الاستكشافي البصري

رسم التوزيع والبوكس بلوت لاكتشاف القيم الشاذة.

```python
sns.histplot(df['chol'], kde=True)
sns.boxplot(x=df['thalch'])
```

---

##  تنظيف البيانات

معالجة القيم المفقودة بالأكثر شيوعًا أو الوسيط.

```python
df['chol'] = df['chol'].fillna(df['chol'].median())
df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
```

---

##  موازنة البيانات

استخدام oversampling لزيادة العينات الإيجابية أو السلبية.

```python
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X, y)
```

---

##  تجهيز البيانات

نقوم بتحويل القيم الفئوية والرقمية وتوحيد القياسات.

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
```

---

##  بناء نموذج الشبكة العصبية

هيكل النموذج يحتوي على 3 طبقات مخفية وواحدة إخراجية.

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=input_dim))
model.add(Dropout(0.4))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

---

##  تدريب النموذج

تجميع النموذج باستخدام خوارزمية SGD وتدريبه لمدة 100 تكرار.

```python
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100)
```

---

##  تقييم النموذج

حساب الدقة والمقاييس المختلفة مع رسم مصفوفة الالتباس.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

y_pred = model.predict(x_test)
y_pred_binary = np.round(y_pred)

print("Accuracy:", accuracy_score(y_test, y_pred_binary))
print("Precision:", precision_score(y_test, y_pred_binary))
print("Recall:", recall_score(y_test, y_pred_binary))
print("F1 Score:", f1_score(y_test, y_pred_binary))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_binary))
```

---

##  حفظ النموذج والـ Encoders

نحفظ النموذج والـ transformers للاستخدام لاحقًا في التنبؤ.

```python
from joblib import dump

model.save('heart_disease.keras')
dump(labelencoder_gender, 'heart_disease_label.pkl')
dump(ct, 'heart_disease_column_transformers.pkl')
dump(sc, 'heart_disease_standard_scaler.pkl')
```

---

##  التنبؤ بحالة مريض جديد

تحميل الـ encoders والنموذج وتحويل البيانات الجديدة والتنبؤ بالحالة.

```python
import pandas as pd
from keras.models import load_model
from joblib import load

model = load_model('heart_disease.keras')
labelencoder_gender = load('heart_disease_label.pkl')
ct_loaded = load('heart_disease_column_transformers.pkl')
sc_loaded = load('heart_disease_standard_scaler.pkl')

new_patient = pd.DataFrame([...], columns=...)

new_patient['sex'] = labelencoder_gender.transform(new_patient['sex'])
new_patient_encoded = ct_loaded.transform(new_patient)
new_patient_scaled = sc_loaded.transform(new_patient_encoded)

prediction = model.predict(new_patient_scaled)
print(prediction)
```
