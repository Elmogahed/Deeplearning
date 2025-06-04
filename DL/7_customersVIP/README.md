# تصنيف العملاء باستخدام الشبكات العصبية الاصطناعية (Neural Networks)

يهدف هذا المشروع إلى بناء نموذج تنبؤي باستخدام **شبكة عصبية اصطناعية** لتصنيف العملاء إلى فئتين (نعم/لا) بناءً على بياناتهم الديموغرافية وسلوكهم خلال الحملات التسويقية.

---

##  فكرة المشروع

المشروع يعتمد على مجموعة من البيانات المتعلقة بالعملاء، مثل العمر، الوظيفة، الحالة الاجتماعية، نوع الاتصال، وغيرها. نستخدم هذه البيانات لتوقع ما إذا كان العميل سيستجيب للحملة الإعلانية أم لا (المتغير `y`).

---

##  البيانات المستخدمة

يتم تحميل البيانات من ملف `customers.csv`، وتحتوي على أعمدة مثل:

- معلومات شخصية: `age`, `job`, `marital`, `education`
- معلومات مالية: `balance`, `housing`, `loan`
- معلومات التواصل: `contact`, `day`, `month`, `duration`
- معلومات الحملة التسويقية: `campaign`, `pdays`, `previous`, `poutcome`
- المتغير الهدف: `y` (نعم/لا)

---

##  خطوات التنفيذ

### 1. تحميل البيانات واستكشافها

```python
df = pd.read_csv('customers.csv')
print(df.head())
print(df.info())
value_counts = df['y'].value_counts()
```

### 2. معالجة عدم التوازن في الفئات

تم استخدام `RandomOverSampler` لتوازن عدد السجلات بين الفئتين:

```python
from imblearn.over_sampling import RandomOverSampler
oversampler = RandomOverSampler(random_state=0)
input_columns_resampled, class_column_resampled = oversampler.fit_resample(input_columns, class_column)
```

---

### 3. معالجة البيانات (الترميز والتحويل)

#### • ترميز المتغير الهدف `y` باستخدام `LabelEncoder`
#### • ترميز الأعمدة الفئوية باستخدام `OneHotEncoder`

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
```

تم تطبيق الترميز على العمود `job` أولاً ثم على باقي الأعمدة الفئوية.

---

### 4. فصل البيانات إلى تدريب واختبار وتوحيد القيم

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

---

##  بناء النموذج باستخدام Keras

تم بناء نموذجين:

###  النموذج الأول:

```python
classifier = Sequential()
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
```

###  النموذج الثاني (مع تحسين Dropout):

```python
classifier = Sequential()
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
```

###  تدريب النموذج:

```python
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifierHistory = classifier.fit(X_train, y_train, epochs = 100, validation_data=(X_test, y_test))
```

---

##  تقييم الأداء

### • رسم دقة التدريب والاختبار:

```python
plt.plot(classifierHistory.history['accuracy'])
plt.plot(classifierHistory.history['val_accuracy'])
```

### • حساب المؤشرات:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```

- **الدقة (Accuracy)**
- **الدقة الإيجابية (Precision)**
- **الاسترجاع (Recall)**
- **معدل F1 (F1 Score)**

---

##  النتائج المتوقعة

قد تختلف النتائج حسب البيانات، لكن كمثال:

```
Accuracy: 91.25
Precision: 90.47
Recall: 89.01
F1 Score: 89.73
```

---

##  المتطلبات

لتشغيل المشروع، تأكد من تثبيت الحزم التالية:

```bash
pip install pandas numpy scikit-learn imbalanced-learn tensorflow matplotlib
```
