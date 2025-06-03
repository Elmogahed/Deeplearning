مشروع تصنيف الرسائل (سبام أو لا) باستخدام الشبكات العصبية
هذا المشروع يستخدم بيانات البريد الإلكتروني لبناء نموذج تصنيف الرسائل إلى "Spam" أو "Not Spam" باستخدام معالجة اللغة الطبيعية (NLP) وشبكة عصبية مبنية بـ Keras.

المتطلبات
Python 3.7 أو أحدث

المكتبات:

pandas

nltk

matplotlib

wordcloud

scikit-learn

tensorflow / keras

تثبيت المكتبات:

pip install pandas nltk matplotlib wordcloud scikit-learn tensorflow
الخطوات
تحميل البيانات من ملف emails.csv

معالجة اللغة الطبيعية:

تحويل النص إلى حروف صغيرة

إزالة علامات الترقيم

حذف كلمات التوقف (Stopwords)

استخدام Stemming (PorterStemmer)

تصور كلمات الرسائل المزعجة باستخدام WordCloud

تحويل النص إلى ميزات عددية باستخدام:

CountVectorizer

TfidfVectorizer

تقسيم البيانات إلى تدريب واختبار

بناء نموذج شبكة عصبية باستخدام Keras:

طبقتان مخفيتان: 64 و32 وحدة

طبقة إخراج مع تفعيل sigmoid

تدريب النموذج لمدة 10 عصور (Epochs)

تقييم النموذج باستخدام الدقة (Accuracy)

اختبار النموذج على رسالة جديدة

اختبار على رسالة جديدة

message="call to get free prize one million dollars"
النموذج يصنفها بشكل مباشر كـ "سبام" أو "ليست سبام".

🇬🇧 English
Spam Classification Using Neural Networks
This project builds a neural network model to classify emails as spam or not spam using text preprocessing and vectorization.

Requirements
Python 3.7+

Libraries:

pandas

nltk

matplotlib

wordcloud

scikit-learn

tensorflow / keras

Install via:

pip install pandas nltk matplotlib wordcloud scikit-learn tensorflow
Steps
Load dataset from emails.csv

Text preprocessing:

Lowercasing

Remove punctuation

Remove stopwords

Stemming

Visualize spam words using WordCloud

Text vectorization:

CountVectorizer

TfidfVectorizer

Train/test split

Neural network using Keras:

2 hidden layers (64, 32 units)

Output layer with sigmoid

Train the model

Evaluate using accuracy

Test with new message

Predict New Message

message="call to get free prize one million dollars"
Model classifies it as spam or not spam.

Technical Summary
Data: emails.csv with columns Message and Spam

Text Preprocessing:

word_tokenize, stopwords, PorterStemmer, re (for links and digits)

Vectorization:

CountVectorizer(max_features=100)

TfidfVectorizer(max_features=100)

Model:

Sequential([Dense(64), Dense(32), Dense(1)])

activation='relu' for hidden layers

activation='sigmoid' for output

Loss: binary_crossentropy

Optimizer: adam

Metric: accuracy

Final Test: Real-time input classification using model.predict

