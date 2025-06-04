#  مقارنة أداء CPU و GPU في تدريب شبكة عصبية

##  المتطلبات

```bash
pip install tensorflow numpy
```

---

##  فكرة المشروع

يهدف المشروع إلى مقارنة زمن تدريب نموذج شبكة عصبية باستخدام مكتبة TensorFlow على **المعالج المركزي (CPU)** مقابل **معالج الرسومات (GPU)**. يتم بناء نموذج بسيط يحتوي على عدة طبقات مخفية وتدريبه على بيانات عشوائية.

---

## بناء نموذج الشبكة العصبية

يتم بناء نموذج من نوع `Sequential` يحتوي على 5 طبقات مخفية وتفعيلات `ReLU`، وطبقة إخراجية واحدة بتفعيل `sigmoid`.

```python
model = Sequential()
model.add(Input(shape=(10,)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
```

---

##  إعداد البيانات

يتم توليد بيانات تدريب عشوائية تحتوي على 10 خصائص وعدد 10,000 عينة.

```python
x_train = np.random.rand(10000, 10)
y_train = np.random.randint(2, size=(10000, 1))
```

---

## تدريب النموذج ومقارنة الأداء

###  على المعالج المركزي (CPU)

```python
with tf.device('/CPU:0'):
    model.fit(x=x_train, y=y_train, epochs=10000, batch_size=32, verbose=0)
```

### على معالج الرسومات (GPU)

```python
with tf.device('/GPU:0'):
    model.fit(x=x_train, y=y_train, epochs=10000, batch_size=32, verbose=0)
```

###  حساب الزمن المستغرق

```python
print('Training Time on CPU: {:.4f} seconds'.format(cpu_time))
print('Training Time on GPU: {:.4f} seconds'.format(gpu_time))
```

---

##  المخرجات المتوقعة

يتم طباعة الزمن المستغرق لتدريب النموذج على كل من CPU وGPU، لتقييم الأداء ومقارنة السرعة.

```
Training Time on CPU: 45.3214 seconds
Training Time on GPU: 12.6547 seconds
```

---

##  ملاحظات

- تأكد من أن جهازك يحتوي على **معالج رسومات مدعوم من TensorFlow** (مثل NVIDIA مع تعريفات CUDA).
- بعض البيئات الافتراضية أو الأجهزة قد لا تدعم GPU، وسيتم استخدام CPU افتراضيًا.
