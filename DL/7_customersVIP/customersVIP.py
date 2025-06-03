import numpy as np
import pandas as pd

df = pd.read_csv('customers.csv')
print(df.head())
print(df.info())
value_counts = df['y'].value_counts()
print(value_counts)

from imblearn.over_sampling import RandomOverSampler
input_columns = df.drop('y', axis=1)
class_column = df['y']
oversampler = RandomOverSampler(random_state=0)
input_columns_resampled, class_column_resampled = oversampler.fit_resample(input_columns, class_column)
df = pd.concat([input_columns_resampled, class_column_resampled], axis=1)
class_distribution = df['y'].value_counts()
print(class_distribution)

X = df.iloc[:,0:16].values
y = df.iloc[:,-1]

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(y)

X_job = X[:,[1]]
print(np.unique(X_job))

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],
                       sparse_threshold=0)
X_job = ct.fit_transform(X_job)
print(X_job.shape)
print(X_job)
X_cat = X[:,[1, 2, 3, 4, 6, 7, 8, 10, 15]]
print(X_cat.shape)
orginalNumOfCols = X_cat.shape[1]

for i in range(X_cat.shape[1]):
    currNumOfCols = X_cat.shape[1]

    indexOfColumnToEncode = currNumOfCols - orginalNumOfCols + i

    ct = ColumnTransformer(transformers=
                           [('encoder',
                             OneHotEncoder(), [indexOfColumnToEncode])],
                           remainder='passthrough',
                           sparse_threshold=0)

    X_cat = ct.fit_transform(X_cat)

print(X_cat.shape)
X_num = X[:,[0, 5, 9, 11, 12, 13, 14]]

X = np.concatenate((X_num,X_cat), axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 64, activation = 'relu'))

classifier.add(Dense(units = 32, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifierHistory = classifier.fit(X_train, y_train, epochs = 100, validation_data=(X_test, y_test))

from keras.layers import Dropout

classifier = Sequential()

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 64, activation = 'relu'))

classifier.add(Dropout(0.5))

classifier.add(Dense(units = 32, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifierHistory = classifier.fit(X_train, y_train, epochs = 100, validation_data=(X_test, y_test))

import matplotlib.pyplot as plt

plt.plot(classifierHistory.history['accuracy'])
plt.plot(classifierHistory.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
evaluation = classifier.evaluate(X_test, y_test)

print("Loss:", evaluation[0])
print("Accuracy:", evaluation[1])
y_pred = classifier.predict(X_test)

y_pred_binary = (y_pred > 0.5)

print(y_pred_binary)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", round(100*accuracy,2))

precision = precision_score(y_test, y_pred_binary)
print("Precision:", round(100*precision,2))

recall = recall_score(y_test, y_pred_binary)
print("Recall:", round(100*recall,2))

f1 = f1_score(y_test, y_pred_binary)
print("F1 Score:", round(100*f1,2))