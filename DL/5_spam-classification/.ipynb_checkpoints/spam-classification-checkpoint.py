import pandas as pd

df = pd.read_csv('emails.csv')
print(df.head())

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


text="Hi: I am working on spam classification!"

tokens = word_tokenize(text.lower())
print(tokens)

import string

tokens = [token for token in tokens if token not in string.punctuation]
print(tokens)

print(string.punctuation)

stop_words = stopwords.words('english')

tokens = [token for token in tokens if token not in stop_words]
print(tokens)

print(stop_words)
print(len(stop_words))

stemmer = PorterStemmer()
tokens = [stemmer.stem(token) for token in tokens]
print(tokens)

ps = PorterStemmer()

print(ps.stem("computer"))
print(ps.stem("computation"))
print(ps.stem("compute"))
print(ps.stem("computed"))
print(ps.stem("computers"))

import re

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation]
    stop_words = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    preprocess_text = ' '.join(tokens)
    preprocess_text = re.sub(r'http\S+|www\S+', '', preprocess_text)
    preprocess_text = re.sub(r'\d+', '', preprocess_text)
    return preprocess_text

email = "Hello! These are 5678 examples of emails for spam classification. "
preprocessed_email = preprocess_text(email)
print(preprocessed_email)

df['processed_Message'] = df['Message'].apply(preprocess_text)
print(df)
df_spam = df[df['Spam']==1]
import matplotlib.pyplot as plt

from wordcloud import WordCloud

spam_words_list = df_spam['processed_Message'].astype(str)
spam_words_str = ' '.join(spam_words_list)
spam_wordcloud = WordCloud(width=500, height=300).generate(spam_words_str)
plt.figure(figsize=(6, 4))
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.show()

from sklearn.feature_extraction.text import CountVectorizer

documnets = [
    "My cat and dog sat on the mat",
    "strange cat jumped over the mat",
    "His cat ate the mouse ran across the mat",
    "Her dog run after my cat and mouse",
]
max_features = 6
count_vectorizer = CountVectorizer(max_features=max_features)
vectors = count_vectorizer.fit_transform(documnets)

print(vectors)

feature_names = count_vectorizer.get_feature_names_out()
print('feature_names')
print(feature_names)
print(vectors.toarray())

example_df = pd.DataFrame(data=vectors.toarray(), columns=feature_names)
print(example_df)

frequencies = vectors.toarray().sum(axis=0)
sorted_indices = frequencies.argsort()[::-1]
sorted_feature_names = [feature_names[idx] for idx in sorted_indices]
sorted_frequencies = frequencies[sorted_indices]

for i in range(len(sorted_feature_names)):
    print(sorted_feature_names[i], ": ", sorted_frequencies[i])

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
tfidf_vectors = tfidf_vectorizer.fit_transform(documnets)

print("\nTF-IDF Vectorization:")

feature_names=tfidf_vectorizer.get_feature_names_out()
example_df = pd.DataFrame(data=tfidf_vectors.toarray(),columns=feature_names)
print(example_df)


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = df['processed_Message']


max_features = 100

count_vectorizer = CountVectorizer(max_features=max_features)

vectors = count_vectorizer.fit_transform(corpus)

tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
tfidf_vectors = tfidf_vectorizer.fit_transform(corpus)

from sklearn.model_selection import train_test_split

X = vectors

y = df['Spam']

X = X.toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)

import numpy as np

y_pred_prob = model.predict(X_test)
y_pred = np.round(y_pred_prob)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy*100,2))

from sklearn.model_selection import train_test_split

X = tfidf_vectors

y = df['Spam']

X = X.toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)

import numpy as np

y_pred_prob = model.predict(X_test)
y_pred = np.round(y_pred_prob)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy*100,2))

message="call to get free prize one million dollars"

processed_message=preprocess_text(message)

vector=tfidf_vectorizer.transform([processed_message])
vector_dense=vector.toarray()

y_pred_prob = model.predict(vector_dense)

y_pred = np.round(y_pred_prob)
print(y_pred)

