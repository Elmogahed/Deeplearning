
# مشروع الانحدار الخطي باستخدام Keras و TensorFlow

هذا المشروع يعرض كيفية إنشاء نموذج انحدار خطي باستخدام مكتبة Keras ضمن بيئة TensorFlow، ويتضمن عدة طرق لتعريف النموذج وتدريبه وقياس الأداء بالإضافة إلى رسم منحنى الخطأ.

---

##  خطوات المشروع

### 1. استيراد المكتبات اللازمة

```python
from keras.models import Sequential
from keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt
```

---

### 2.  إنشاء النموذج وتكوينه

```python
model = Sequential()
model.add(Input(shape=(1,)))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer='sgd', loss='mean_absolute_error', metrics=['mae', 'mape'])
```

- تعريف نموذج بسيط مكون من طبقة واحدة.
- اختيار دالة خسارة `mean_absolute_error`.
- استخدام المحسن `SGD` (الانحدار التدرجي).

---

### 3.  تدريب النموذج

```python
x_train = np.array([1, 2, 3, 4])
y_train = np.array([2, 4, 6, 8])
model.fit(x=x_train, y=y_train, epochs=1000, batch_size=2)
```

- البيانات تمثل علاقة خطية: `y = 2x`.

---

### 4.  استخراج النتائج

```python
weights, bias = model.get_weights()
print("Updated weights and bias:", weights, bias)
```

- طباعة القيم النهائية للوزن والانحياز بعد التدريب.

---

### 5. التنبؤ بقيم جديدة

```python
preds = model.predict(np.array([5,6,7]))
print(preds)
```

- توقع نتائج جديدة بعد التدريب.

---

### 6.  تجربة نماذج مختلفة

تم تكرار التدريب عدة مرات بتغييرات طفيفة مثل:

- دمج `input_shape` مع الطبقة `Dense` مباشرة.
- استخدام `mean_squared_error` بدلًا من `mean_absolute_error`.
- استخدام `optimizer` مع معدل تعلم معين.

---

### 7.  رسم منحنى الخطأ

```python
history = model.fit(...)
plt.plot(range(epochs), history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
```

- عرض تطور قيمة الخسارة عبر عدد التكرارات (epochs).

---

### 8. رسم منحنى MAE

```python
plt.plot(range(epochs), history.history['mae'])
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('Training MAE')
```

- عرض تطور قيمة الخطأ المطلق المتوسط (MAE).

---

## ✅ المتطلبات

```bash
pip install tensorflow matplotlib numpy
```

---

