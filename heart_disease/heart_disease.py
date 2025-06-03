# Import libraries and read data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('heart_disease_uci.csv')
print(df.head())
print(df.info())
print(df.duplicated().sum())
print(df.isnull().sum().sum())
print(df.isnull().sum())

cols = ['trestbps', 'chol', 'thalch', 'oldpeak']
fig , axes = plt.subplots(2, 2 ,figsize=(12,8))
axes = axes.flatten()

for i, col in enumerate(cols):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

cols = ['trestbps', 'chol', 'thalch', 'oldpeak']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(cols):
    sns.boxplot(x=df[col], ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()





# Handling null values
df['trestbps'] = df['trestbps'].fillna(df['trestbps'].median())
df['chol']  = df['chol'].fillna(df['chol'].median())
df['thalch'] = df['thalch'].fillna(df['thalch'].median())
df['oldpeak'] = df['oldpeak'].fillna(df['oldpeak'].median())

df['fbs'] = df['fbs'].fillna(df['fbs'].mode()[0])
df['restecg'] = df['restecg'].fillna(df['restecg'].mode()[0])
df['exang'] = df['exang'].fillna(df['exang'].mode()[0])
df['slope'] = df['slope'].fillna(df['slope'].mode()[0])

df['ca'] = df['ca'].fillna(df['ca'].median())
df['thal'] = df['thal'].fillna(df['thal'].mode()[0])

print(df.isnull().sum())
print(df.info())

# Data balancing
value_counts = df['num'].value_counts()
print(value_counts)
labels = value_counts.index
counts = value_counts.values
plt.pie(counts, labels= labels, autopct='%1.1f%%')
plt.show()
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
value_counts = df['num'].value_counts()
labels = value_counts.index
counts = value_counts.values
plt.pie(counts, labels=labels, autopct='%1.1f%%')
plt.show()

from imblearn.over_sampling import RandomOverSampler
input_columns = df.drop('num', axis=1)
class_columns = df['num']
oversampler = RandomOverSampler(random_state=0)
input_columns_resampled , class_columns_resampled = oversampler.fit_resample(input_columns, class_columns)
df = pd.concat([input_columns_resampled, class_columns_resampled], axis=1)
class_distribution = df['num'].value_counts()
print(class_distribution)

value_counts = df['num'].value_counts()
labels = value_counts.index
counts = value_counts.values
plt.pie(counts, labels= labels, autopct='%1.1f%%')
plt.show()



x = df.drop(columns=['id', 'num'])
print(x)
y = df['num']
print(y)

print(x['sex'])

from sklearn.preprocessing import LabelEncoder
labelencoder_gender = LabelEncoder()
x['sex'] = labelencoder_gender.fit_transform(x['sex'])

print(x['sex'])
print(labelencoder_gender.transform(['Male', 'Female']))

x['fbs'] = x['fbs'].astype(int)
x['exang']  = x['exang'].astype(int)

distinct_values  = np.unique(x['dataset'])
print(distinct_values)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

multi_class_columns = ['dataset', 'cp', 'restecg', 'slope', 'thal']
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), multi_class_columns)], remainder='passthrough')
x = ct.fit_transform(x)
print(x[0])


from sklearn.model_selection import train_test_split
x_train , x_val_and_test , y_train, y_val_and_test = train_test_split(x,y, test_size =0.30, random_state=42)
x_val, x_test , y_val , y_test = train_test_split(x_val_and_test, y_val_and_test, test_size=0.50, random_state=42)
print('X_train.shape:', x_train.shape)
print('X_val.shape  :', x_val.shape)
print('X_test.shape :', x_test.shape)
print('Y_train.shape:', y_train.shape)
print('Y_val.shape  :', y_val.shape)
print('Y_test.shape :', y_test.shape)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_val = sc.transform(x_val)
x_test = sc.transform(x_test)

print(x_train)

import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers

model = Sequential()

input_dim = len(x_train[0])
model.add(Dense(32, activation='relu', input_dim= input_dim , kernel_regularizer=regularizers.l2(0.00001)))
model.add(Dropout(0.4)) 
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
model.add(Dropout(0.4)) 
model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

history =  model.fit(x_train, y_train, batch_size=20, epochs=100, validation_data=(x_val, y_val))
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Valdation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


evaluation = model.evaluate(x_test, y_test)

print("Loss:", evaluation[0])
print("Accuracy:", evaluation[1])

y_pred = model.predict(x_test)
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

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_binary)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

model.save('heart_disease.keras')
from joblib import dump

dump(labelencoder_gender, 'heart_disease_label.pkl')
dump(ct, 'heart_disease_column_transformers.pkl')
dump(sc,'heart_disease_standard_scaler.pkl')

# new_patient = np.array([[70, 'Male', 'Cleveland', 'asymptomatic', 160, 286,  False, 'lv hypertrophy' , 108, True, 1.5, 'flat', 3, 'normal']])
from joblib import load
labelencoder_gender_loaded = load('heart_disease_label.pkl')
# new_patient[:, 1] = labelencoder_gender_loaded.transform(new_patient[:,1])

ct_loaded = load('heart_disease_column_transformers.pkl')
# new_patient = ct_loaded.transform(new_patient)
sc_loaded = load('heart_disease_standard_scaler.pkl')
# new_patient = sc_loaded.transform(new_patient)
# print(new_patient)

columns = ['age', 'sex', 'dataset', 'cp', 'trestbps', 'chol', 'fbs',
           'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
new_patient = pd.DataFrame([[70, 'Male', 'Cleveland', 'asymptomatic', 160, 286,
                              False, 'lv hypertrophy', 108, True, 1.5, 'flat', 3, 'normal']],
                           columns=columns)
# نفس العمليات السابقة
new_patient['sex'] = labelencoder_gender_loaded.transform(new_patient['sex'])
new_patient['fbs'] = new_patient['fbs'].astype(int)
new_patient['exang'] = new_patient['exang'].astype(int)

new_patient = ct_loaded.transform(new_patient)
new_patient = sc_loaded.transform(new_patient)
print(new_patient)

from keras.models import load_model
loaded_model = load_model('heart_disease.keras')

new_prediction_proba = loaded_model.predict(new_patient)
new_prediction = (new_prediction_proba > 0.5)
print(new_prediction)

