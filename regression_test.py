import tensorflow as tf
import numpy as np
import pylab


x_data = np.random.rand(100).astype(np.float32)
noise = np.random.normal(scale=0.01, size=len(x_data))
y_data = x_data * 0.1 + 0.3 + noise

# pylab.plot(x_data, y_data, '.')
# pylab.show()

print("input data has been generated")

W = tf.Variable(tf.random_uniform([1], 0.0, 1.0), name="weight")
B = tf.Variable(tf.zeros([1]), name="bias")
y = W * x_data + B

print(W)
print(B)

# Build training graph
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss=loss)
init = tf.global_variables_initializer()

print("Loss: ", loss)
print("Optimizer: ", optimizer)
print("Train: ", train)
print("Init: ", init)

print(tf.get_default_graph().as_graph_def())

sess = tf.Session()
sess.run(init)
y_initial_value = sess.run(y)

print(sess.run([W, B]))

for step in range(201):
    sess.run(train)

print(sess.run([W, B]))


pylab.plot(x_data, y_data, '.', label="target_value")
pylab.plot(x_data, y_initial_value, '.', label="initial_values")
pylab.plot(x_data, sess.run(y), '.', label="trained_values")
pylab.legend()
pylab.ylim(0, 1.0)
pylab.show()
