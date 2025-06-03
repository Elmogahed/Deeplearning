import pandas as pd
df = pd.read_csv('Churn_Modelling.csv')
print(df.head())

# موازنة الأصناف كي لا ينحاز النموذج إلى الصنف الأكثر
value_counts = df['Exited'].value_counts()
print(value_counts)

import matplotlib.pyplot as plt
labels = value_counts.index
counts = value_counts.values
plt.pie(counts, labels= labels, autopct='%1.1f%%')
plt.title('Distribution of Exited')
plt.show()

# إضافة أمثلة عشوائية للصنف الأقل لموزانة الأصناف
from imblearn.over_sampling import RandomOverSampler
input_columns = df.drop('Exited', axis=1)
class_column = df['Exited']
oversampler = RandomOverSampler(random_state=0)
input_columns_resampled, class_column_resampled = oversampler.fit_resample(input_columns, class_column)
df_balanced = pd.concat([input_columns_resampled, class_column_resampled], axis=1)
class_distribution = df_balanced['Exited'].value_counts()
print(class_distribution)

value_counts = df_balanced['Exited'].value_counts()
labels = value_counts.index
counts = value_counts.values
plt.pie(counts, labels=labels, autopct='%1.1f%%')
plt.title('Distribution of Exited')
plt.show()

X = df_balanced.iloc[:, 3:13].values
print(X)

y = df_balanced.iloc[:, 13].values
print(y)
print(X[:, 2])

from sklearn.preprocessing import LabelEncoder
labelencoder_gender = LabelEncoder()
X[:, 2] = labelencoder_gender.fit_transform(X[:, 2])
print(X[:, 2])
print(labelencoder_gender.transform(["Male","Female"]))

import numpy as np
distinct_values = np.unique(X[:,1])
print(distinct_values)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)
print(X[0])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)

import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
print("==============================")
print(len(X_train[0]))
print("==============================")
input_dim = len(X_train[0])
model.add(Dense(6, activation = 'relu', input_dim = input_dim))
model.add(Dense(6, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print(model.summary())
model.fit(X_train, y_train, batch_size = 10, epochs = 10)
evaluation = model.evaluate(X_test, y_test)
print("Loss:", evaluation[0])
print("Accuracy:", evaluation[1])

y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5)
print(y_pred_binary)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)
precision = precision_score(y_test, y_pred_binary)
print("Precision:", precision)
recall = recall_score(y_test, y_pred_binary)
print("Recall:", recall)
f1 = f1_score(y_test, y_pred_binary)
print("F1 Score:", f1)

model = Sequential()
input_dim = len(X_train[0])
model.add(Dense(128, activation = 'relu', input_dim = input_dim))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy' ])
model.fit(X_train, y_train, batch_size = 10, epochs = 20)


y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5)
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)
precision = precision_score(y_test, y_pred_binary)
print("Precision:", precision)
recall = recall_score(y_test, y_pred_binary)
print("Recall:", recall)
f1 = f1_score(y_test, y_pred_binary)
print("F1 Score:", f1)


model.save("churn_model.keras")

from joblib import dump

dump(labelencoder_gender, "churn_label_encoder.pkl")
dump(ct, "churn_column_transformer.pkl")
dump(sc, "churn_standard_scaler.pkl")
new_customer = np.array( [[600,"France","Male",40, 3, 60000, 2, 1, 1, 50000]])
from joblib import load
labelencoder_gender_loaded=load("churn_label_encoder.pkl")
new_customer[:, 2] = labelencoder_gender_loaded.transform(new_customer[:, 2])
ct_loaded = load("churn_column_transformer.pkl")
new_customer = ct.transform(new_customer)
sc_loaded = load("churn_standard_scaler.pkl")
new_customer = sc_loaded.transform(new_customer)
print(new_customer)
from keras.models import load_model
loaded_model = load_model("churn_model.keras")
new_prediction_proba = loaded_model.predict(new_customer)
new_prediction = (new_prediction_proba > 0.5)
print(new_prediction)