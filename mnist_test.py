import math
import os
from six.moves import xrange
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import matplotlib.pyplot as plt

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
BATCH_SIZE = 100
EVAL_BATCH_SIZE = 1

HIDDEN1_UNITS = 128
HIDDEN2_UNITS = 32

MAX_STEPS = 2000

TRAIN_DIR = "/tmp/mnist"

data_sets = read_data_sets(TRAIN_DIR, False)


def mnist_inference(images, hidden1_units, hidden2_units):
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units], stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))), name="weights")
        biases = tf.Variable(tf.zeros([hidden1_units]), name="biases")
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0 / math.sqrt(float(hidden1_units))), name="weights")
        biases = tf.Variable(tf.zeros([hidden2_units]), name="biases")
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    with tf.name_scope('soft_max_layer'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES], stddev=1.0/math.sqrt(float(hidden2_units))),
            name="weights"
        )

        biases = tf.Variable(
            tf.zeros([NUM_CLASSES]),
            name="biases"
        )

        logits1 = tf.matmul(hidden2, weights) + biases
        tf.train.write_graph(tf.get_default_graph().as_graph_def(), "/tmp/mnist", "mnist_inference_graph.pbtxt", as_text=True)
    return logits1


def mnist_training(logits2, labels, learning_rate):
        labels = tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=labels, name="xentropy")
        loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="GradientDescent")
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op, loss


mnist_graph = tf.Graph()
with mnist_graph.as_default():
    images_placeholder = tf.placeholder(tf.float32)
    labels_placeholder = tf.placeholder(tf.int32)
    tf.add_to_collection("images", images_placeholder)
    tf.add_to_collection("labels", labels_placeholder)

    logits = mnist_inference(images_placeholder, HIDDEN1_UNITS, HIDDEN2_UNITS)
    tf.add_to_collection("logits", logits)

    train_op, loss = mnist_training(logits, labels_placeholder, 0.01)

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()
    tf.train.write_graph(tf.get_default_graph().as_graph_def(), logdir="/tmp/mnist", name="graph_structure", as_text=True)

with tf.Session(graph=mnist_graph) as sess:
    sess.run(init)

    losses = []
    for step in xrange(MAX_STEPS):
        images_feed, labels_feed = data_sets.train.next_batch(BATCH_SIZE)
        _, loss_value = sess.run([train_op, loss], feed_dict={images_placeholder: images_feed,
                                                              labels_placeholder: labels_feed})

        losses.append(loss_value)
        if step % 1000 == 0:
            print('Step %d, loss=%.2f' % (step, loss_value))

    plt.plot(losses)
    plt.show()
    checkpoint_file = os.path.join(TRAIN_DIR, 'checkpoint')
    saver.save(sess, checkpoint_file, global_step=step)

with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(os.path.join(TRAIN_DIR,"checkpoint-1999.meta"))
    saver.restore(sess, os.path.join(TRAIN_DIR, "checkpoint-1999"))

    logits = tf.get_collection('logits')[0]
    images_placeholder = tf.get_collection('images')[0]
    labels_placeholder = tf.get_collection('labels')[0]

    eval_op = tf.nn.top_k(logits)

    images_feed, labels_feed = data_sets.validation.next_batch(EVAL_BATCH_SIZE)
    imgplot = plt.imshow(np.reshape(images_feed, (28, 28)))
    prediction = sess.run(eval_op, feed_dict={images_placeholder: images_feed,
                                              labels_placeholder: labels_feed})
    print("Ground truth: %d\nPrediction: %d" % (labels_feed, prediction.indices[0][0]))


