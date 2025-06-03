import numpy as np
from keras.models import Sequential
from keras.layers import Dense

x = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1]])
y = np.array([[0],[1],[0], [1],[0], [1]])

model = Sequential()
model.add(Dense(input_shape=(3,), units=4, activation='relu'))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(x,y, epochs=300, batch_size=2)
loss , accuracy = model.evaluate(x,y)
print(accuracy)

y_pred = model.predict(x)
print(y_pred)

threshold = 0.5
y_pred_class = (y_pred  > threshold).astype(int)
print(y_pred_class)

import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss = history.history['loss']
plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.plot(acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')

plt.subplot(1,2,2)
plt.plot(loss, label = 'Training Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()
