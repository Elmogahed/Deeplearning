import pandas as pd

df = pd.read_csv('housepricedata.csv')
pd.set_option('display.max_columns', None)
print(df.head())
print(df.shape)

X = df.values[:,0:10]
Y = df.values[:,10]

from sklearn.preprocessing import MinMaxScaler

min_max_scaler =  MinMaxScaler()
X_Scale = min_max_scaler.fit_transform(X)
print(X_Scale)

from sklearn.model_selection import train_test_split

X_train , X_val_and_test , Y_train, Y_val_and_test = train_test_split(X_Scale, Y, test_size=0.3)
X_val , X_test ,Y_val ,Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

print("X_train.shape : ",X_train.shape)
print("X_test.shape : ", X_test.shape)
print("X_val.shape : ", X_val.shape)
print("Y_val.shape : ", Y_val.shape)
print("Y_train.shape : ", Y_train.shape)
print("Y_test.shape : ", Y_test.shape)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(32, activation= 'relu', input_shape= (10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')])

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))

test_loss ,test_accuracy = model.evaluate(X_test, Y_test)
print(test_loss)
print(test_accuracy)

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper right')
plt.show()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])

plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


model_2 = Sequential([
    Dense(1000, activation='relu', input_shape=(10,)),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1, activation='sigmoid'),
])
model_2.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics = ['accuracy'])
hist_2 = model.fit(X_train,Y_train,
            batch_size=32, epochs= 100,
            validation_data=(X_val, Y_val))
plt.plot(hist_2.history['loss'])
plt.plot(hist_2.history['val_loss'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(hist_2.history['accuracy'])
plt.plot(hist_2.history['val_accuracy'])

plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

from keras.layers import Dropout
from keras import regularizers
model_3 = Sequential([
    Dense(1000, activation='relu', kernel_regularizer= regularizers.l2(0.01), input_shape=(10,)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer= regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer= regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),
])

model_3.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
hist_3 = model_3.fit(X_train, Y_train, batch_size = 32, epochs= 100,
                     validation_data=(X_val, Y_val))

plt.plot(hist_3.history['loss'])
plt.plot(hist_3.history['val_loss'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.ylim(top=1.2, bottom=0)
plt.show()

plt.plot(hist_3.history['accuracy'])
plt.plot(hist_3.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()
from sklearn.metrics import recall_score, f1_score, precision_score
y_pred = model.predict(X_test)

y_pred_classes=[]

for prob in y_pred:
    if prob >= 0.5:
        y_pred_classes.append(1)
    else:
        y_pred_classes.append(0)

precision = precision_score(Y_test, y_pred_classes)

recall = recall_score(Y_test, y_pred_classes)

f1 = f1_score(Y_test, y_pred_classes)

print("Precision:", round(precision*100,0))
print("Recall:", round(recall*100,0))
print("F1 Score:", round(f1*100,0))

from sklearn.metrics import confusion_matrix
import seaborn as sns

confusion_mat = confusion_matrix(Y_test, y_pred_classes)

sns.heatmap(confusion_mat, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

model.save("houses_model.keras")

from joblib import dump

dump(min_max_scaler, "houses_min_max_scaler.pkl")

import numpy as np

new_house = np.array( [[100,6,6,800, 2, 2, 3, 7, 1, 500]])
from joblib import load

min_max_scaler_loaded = load("houses_min_max_scaler.pkl")

new_house = min_max_scaler.transform(new_house)
print(new_house)

from keras.models import load_model
loaded_model = load_model("houses_model.keras")

new_prediction_proba = loaded_model.predict(new_house)
new_prediction = (new_prediction_proba > 0.5)
print(new_prediction)