# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops import gen_math_ops

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)
NB_INSTANTIA_PARAMS = 3
NB_HIDDEN1_UNITS = 10
NB_HIDDEN2_UNITS = 20
# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

def dense_layer(_x, nb_last_units, nb_this_units):
    # 产生一个全连接层
    weights = tf.Variable(tf.truncated_normal([nb_last_units, nb_this_units],
                                              stddev=1.0 / np.sqrt(float(nb_last_units))))
    biases = tf.Variable(tf.zeros([nb_this_units]))     # 偏置
    #layer = tf.nn.sigmoid(tf.matmul(_x, weights) + biases)
    #return layer
    return tf.matmul(_x, weights) + biases


def capsule(images, delta):
    # 产生一个胶囊
    hidden1 = tf.nn.sigmoid(
        dense_layer(images, n_input, NB_HIDDEN1_UNITS))   # 首先是第一个隐层，连接图片和胶囊

    # 然后是第二个隐层，把图片转化成某种高层次表达（encoder）
    hidden2 = tf.nn.sigmoid(
        dense_layer(hidden1, NB_HIDDEN1_UNITS, NB_INSTANTIA_PARAMS))

    # x, y, p
    p = tf.reshape(hidden2[:, 2], shape=[1, batch_size])     # 这个胶囊的权重
    p2 = hidden2[:, 2]
    p3 = tf.reshape(hidden2[:, 2], shape=[batch_size,1])

    instantiation_params = hidden2[:, 0:2]  # 高层次表达实例化参数（这里是x，y坐标）
    #instantiation_params += delta       # 对其应用某种改变（这里是平移）

    # 重新生成图片（decoder）
    hidden3 = tf.nn.sigmoid(
        dense_layer(instantiation_params, NB_INSTANTIA_PARAMS - 1, NB_HIDDEN2_UNITS))

    # 利用p重新计算输出图片
    #capsule_output_images = tf.matmul(p, tf.nn.sigmoid(dense_layer(hidden3, NB_HIDDEN2_UNITS, IMAGE_PIXELS)))
    #capsule_output_images = p2 * tf.nn.sigmoid(dense_layer(hidden3, NB_HIDDEN2_UNITS, IMAGE_PIXELS))

    #capsule_output_images = tf.scalar_mul(p2,tf.nn.sigmoid(dense_layer(hidden3, NB_HIDDEN2_UNITS, IMAGE_PIXELS)))
    capsule_output_images = gen_math_ops.mul(p3,tf.nn.sigmoid(dense_layer(hidden3, NB_HIDDEN2_UNITS, IMAGE_PIXELS)))

    return capsule_output_images



# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()