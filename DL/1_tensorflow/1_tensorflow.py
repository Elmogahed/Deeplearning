import tensorflow as tf
from keras.src.ops import dtype

a = tf.constant(5.0)
b = tf.constant(6.0)

c = a * b
print(c)
print(c.numpy())
a = tf.Variable([0,0], dtype=tf.float32)
b = tf.Variable([0,0], dtype=tf.float32)
a.assign([2,4])
b.assign([5,6])
mul = a * b
output = mul.numpy()
print('Multiplying a and b : ', output)

x = tf.constant([1,2,3,4], dtype=tf.float32)
y = tf.constant([2,4,6,8], dtype=tf.float32)
w = tf.Variable([0.4],dtype=tf.float32)
b = tf.Variable([-0.4],dtype=tf.float32)
linear_model = w * x + b
error = linear_model - y
squared_error = tf.square(error)
loss = tf.reduce_sum(squared_error)
loss_value = loss.numpy()
print('Loss ', loss_value)

learning_rate = 0.01
optimizer = tf._optimizers.SGD(learning_rate=learning_rate)
epochs = 1000
for i in range(epochs):
    with tf.GradientTape() as tape:
        linear_model = w * x + b
        error = linear_model - y
        squared_error = tf.square(error)
        loss = tf.reduce_sum(squared_error)
        print(loss)
    gradients = tape.gradient(loss, [w,b])
    optimizer.apply_gradients(zip(gradients, [w,b]))

w_results = w.numpy()[0]
b_results = b.numpy()[0]
# print('Update values of w and b', round(w_results,2), round(b_results,2))
print('Update values of w and b', w_results, b_results)
