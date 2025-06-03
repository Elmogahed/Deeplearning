import tensorflow.keras as tk

mnist = tk.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(len(train_images), len(test_images))

print(train_images.shape)
print(train_labels.shape)

import matplotlib.pyplot as plt
import numpy as np
unique_values, value_counts = np.unique(train_labels, return_counts=True)
plt.pie(value_counts, labels=unique_values, autopct='%1.1f%%')
plt.show()

print(train_images[0].shape)
print(train_images[0])

max_4 = 25

images_4 = []

for i in range(len(train_labels)):
    if train_labels[i] == 4:
        images_4.append(train_images[i])

        if len(images_4) == max_4:
            break

rows = 5

cols = 5

plt.figure(figsize=(8, 8))

for i in range(max_4):
    plt.subplot(rows, cols, i + 1)

    plt.xticks([])
    plt.yticks([])

    plt.imshow(images_4[i], cmap=plt.cm.binary)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tk.Sequential([
    tk.layers.Flatten(input_shape=(28, 28)),
    tk.layers.Dense(units=512, activation='relu'),
    tk.layers.Dense(units=256, activation='relu'),
    tk.layers.Dense(units=128, activation='relu'),
    tk.layers.Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions[i])

    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(predicted_label, true_label), color=color)
plt.show()