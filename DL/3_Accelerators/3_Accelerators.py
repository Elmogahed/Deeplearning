import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense
import numpy as np
import time

model = Sequential()

model.add(Input(shape=(10,)))

model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


x_train = np.random.rand(10000, 10)
y_train = np.random.randint(2, size=(10000, 1))

epochs = 10000
batch_size = 32

start_cpu = time.time()
with tf.device('/CPU:0'):
    model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, verbose=0)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

# Train on GPU
start_gpu = time.time()
with tf.device('/GPU:0'):
    model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, verbose=0)
end_gpu = time.time()
gpu_time = end_gpu - start_gpu

print('Training Time on CPU: {:.4f} seconds'.format(cpu_time))
print('Training Time on GPU: {:.4f} seconds'.format(gpu_time))