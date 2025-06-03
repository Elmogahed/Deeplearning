from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
import numpy as np

model = Sequential()
model.add(Input(shape= (1,)))
model.add(Dense(units= 1, activation='linear'))
model.compile(optimizer='sgd', loss='mean_absolute_error' , metrics=['mae','mape'])
x_train = np.array([1,2,3,4])
y_train = np.array([2,4,6,8])
epochs = 1000
batch_size = 2
model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size)
wights, bias = model.get_weights()
print('Update Value of w and b ', wights, bias)
preds = model.predict(np.array([5,6,7]))
print(preds)

model = Sequential()
model.add(Dense(input_shape=(1,), units=1, activation='linear'))
model.compile(optimizer='sgd', loss='mean_absolute_error' , metrics=['mae'])
x_train = np.array([1,2,3,4])
y_train = np.array([2,4,6,8])
epochs = 1000
batch_size = 2
model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size)
wights, bias = model.get_weights()
print('Update Value of w and b ', wights, bias)


model = Sequential()
model.add(Dense(units=1, input_shape=(1,), activation='linear'))
from keras.optimizers import SGD
learning_rate = 0.02
optimizer = SGD(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss ='mean_squared_error' )

x_train = np.array([1,2,3,4])
y_train = np.array([2,4,6,8])
epochs = 1000
batch_size = 2
model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size)

import matplotlib.pyplot as plt

model = Sequential()
model.add(Dense(units=1, input_shape=(1,)))
model.compile(optimizer='sgd', loss='mean_absolute_error' , metrics=['mae','mape'])
x_train = np.array([1,2,3,4])
y_train = np.array([2,4,6,8])
epochs = 1000
batch_size = 2
history = model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, verbose=0)
loss_history= history.history['loss']
plt.plot(range(epochs), loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

loss_history = history.history['mae']
plt.plot(range(epochs), loss_history)
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('Training MAE')
plt.show()
