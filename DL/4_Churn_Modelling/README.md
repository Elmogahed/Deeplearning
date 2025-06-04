
# Churn Prediction using Deep Learning

## نظرة عامة

هذا المشروع يهدف إلى بناء نموذج تنبؤ بانسحاب العملاء (Churn) باستخدام بيانات العملاء وتقنيات التعلم العميق. تم استخدام شبكة عصبية متعددة الطبقات مبنية باستخدام Keras مع معالجة مسبقة للبيانات تشمل الترميز والتقييس وموازنة الأصناف.

---

## 1. قراءة البيانات

```python
import pandas as pd

df = pd.read_csv('Churn_Modelling.csv')
print(df.head())
```

---

## 2. تحليل توزيع الأصناف (Exited)

```python
import matplotlib.pyplot as plt

value_counts = df['Exited'].value_counts()
labels = value_counts.index
counts = value_counts.values
plt.pie(counts, labels=labels, autopct='%1.1f%%')
plt.title('Distribution of Exited')
plt.show()
```

---

## 3. موازنة البيانات باستخدام RandomOverSampler

```python
from imblearn.over_sampling import RandomOverSampler

input_columns = df.drop('Exited', axis=1)
class_column = df['Exited']
oversampler = RandomOverSampler(random_state=0)
input_columns_resampled, class_column_resampled = oversampler.fit_resample(input_columns, class_column)
df_balanced = pd.concat([input_columns_resampled, class_column_resampled], axis=1)
```

---

## 4. معالجة البيانات (تحويل الأعمدة الفئوية وتقييس القيم)

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Gender Encoding
labelencoder_gender = LabelEncoder()
X = df_balanced.iloc[:, 3:13].values
X[:, 2] = labelencoder_gender.fit_transform(X[:, 2])

# Country One-Hot Encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)

# Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

y = df_balanced.iloc[:, 13].values
```

---

## 5. تقسيم البيانات للتدريب والاختبار

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

---

## 6. بناء النموذج (شبكة عصبية بسيطة)

```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(6, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=10, epochs=10)
```

---

## 7. التقييم

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5)

accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
```

---

## 8. تحسين النموذج (شبكة أعمق)

```python
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=10, epochs=20)
```

---

## 9. حفظ النموذج والمكونات المسبقة

```python
model.save("churn_model.keras")

from joblib import dump
dump(labelencoder_gender, "churn_label_encoder.pkl")
dump(ct, "churn_column_transformer.pkl")
dump(sc, "churn_standard_scaler.pkl")
```

---

## 10. التنبؤ لعميل جديد

```python
import numpy as np
from joblib import load
from keras.models import load_model

new_customer = np.array([[600, "France", "Male", 40, 3, 60000, 2, 1, 1, 50000]])
labelencoder_gender = load("churn_label_encoder.pkl")
new_customer[:, 2] = labelencoder_gender.transform(new_customer[:, 2])

ct = load("churn_column_transformer.pkl")
new_customer = ct.transform(new_customer)

sc = load("churn_standard_scaler.pkl")
new_customer = sc.transform(new_customer)

model = load_model("churn_model.keras")
prediction = model.predict(new_customer)
print(prediction > 0.5)
```

---

##  النتائج

- الدقة (Accuracy): `تقاس بعد التقييم`
- الاستدعاء (Recall)، الدقة (Precision)، F1: `حسب التقييم في الخطوة 7 و 8`

---

##  الملفات المحفوظة

- `churn_model.keras` : النموذج المدرب.
- `churn_label_encoder.pkl` : مشفر الجنس.
- `churn_column_transformer.pkl` : المحول للأعمدة الفئوية.
- `churn_standard_scaler.pkl` : المقيس القياسي.

---

##  المتطلبات

```bash
pip install pandas scikit-learn matplotlib imbalanced-learn tensorflow keras joblib
```

---
