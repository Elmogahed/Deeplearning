
# مشروع انحدار خطي باستخدام TensorFlow

هذا المشروع يعرض كيفية تنفيذ نموذج انحدار خطي (Linear Regression) باستخدام مكتبة TensorFlow، مع توضيح العمليات الحسابية الأساسية وتدريب النموذج باستخدام الانحدار التدرجي (Gradient Descent).

---

##  شرح الكود

### 1. عمليات رياضية أولية

```python
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
print(c.numpy())  # 30.0
```

- تعريف عددين ثابتين وضربهم وطباعة النتيجة.

---

### 2.  استخدام tf.Variable وتعديل القيم

```python
a = tf.Variable([0, 0], dtype=tf.float32)
b = tf.Variable([0, 0], dtype=tf.float32)
a.assign([2, 4])
b.assign([5, 6])
mul = a * b
print(mul.numpy())  # [10. 24.]
```

- إنشاء متغيرات يمكن تعديلها.
- إسناد القيم لها وضرب كل عنصر في `a` بالمقابل في `b`.

---

### 3.  إنشاء نموذج انحدار خطي

```python
x = tf.constant([1, 2, 3, 4], dtype=tf.float32)
y = tf.constant([2, 4, 6, 8], dtype=tf.float32)
w = tf.Variable([0.4], dtype=tf.float32)
b = tf.Variable([-0.4], dtype=tf.float32)
```

- `x` هي البيانات المدخلة، و `y` هي النتائج الحقيقية.
- `w` هو الوزن، و `b` هو الانحياز.

---

### 4.  حساب الخطأ (Loss)

```python
linear_model = w * x + b
error = linear_model - y
squared_error = tf.square(error)
loss = tf.reduce_sum(squared_error)
print(loss.numpy())
```

- تطبيق نموذج الانحدار الخطي.
- حساب الخطأ بين التوقعات والنتائج الحقيقية.
- حساب مجموع مربعات الخطأ.

---

### 5.  تدريب النموذج باستخدام Gradient Descent

```python
optimizer = tf._optimizers.SGD(learning_rate=0.01)
epochs = 1000

for i in range(epochs):
    with tf.GradientTape() as tape:
        linear_model = w * x + b
        loss = tf.reduce_sum(tf.square(linear_model - y))
    gradients = tape.gradient(loss, [w, b])
    optimizer.apply_gradients(zip(gradients, [w, b]))
```

- التدريب باستخدام الانحدار التدرجي.
- تحديث `w` و `b` لتقليل الخطأ.

---

### 6. النتيجة النهائية

```python
print('Update values of w and b', w.numpy()[0], b.numpy()[0])
```

- طباعة القيم المحسنة للوزن والانحياز.

---

##  المتطلبات

```bash
pip install tensorflow
```

---

