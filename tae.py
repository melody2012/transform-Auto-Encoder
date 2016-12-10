import tensorflow as tf
import mnist
import numpy as np
from tqdm import tqdm

IMAGE_PIXELS = 28 * 28
NB_CAPSULE = 30
NB_HIDDEN1_UNITS = 10
NB_HIDDEN2_UNITS = 20
NB_INSTANTIA_PARAMS = 3

BATCH_SIZE = 32
NB_TRAIN_EPOCH = 100
LEARNING_RATE = 0.01

def dense_layer(_x, nb_last_units, nb_this_units):
    weights = tf.Variable(
        tf.truncated_normal([nb_last_units, nb_this_units],
                            stddev=1.0 / np.sqrt(float(nb_last_units))))
    biases = tf.Variable(tf.zeros([nb_this_units]))
    layer = tf.nn.relu(tf.matmul(_x, weights) + biases)

    return layer


def capsule(images, delta):
    hidden1 = dense_layer(images, IMAGE_PIXELS, NB_HIDDEN1_UNITS)

    hidden2 = dense_layer(hidden1, NB_HIDDEN1_UNITS, NB_INSTANTIA_PARAMS)

    # x, y, p
    p = hidden2[2]
    instatiation_params = hidden2[0:2]
    instatiation_params[0] += delta[0]
    instatiation_params[1] += delta[1]

    hidden3 = dense_layer(instatiation_params, NB_INSTANTIA_PARAMS - 1, NB_HIDDEN2_UNITS)

    capsule_output_images = dense_layer(hidden3, NB_HIDDEN2_UNITS, IMAGE_PIXELS) * p

    return capsule_output_images

def loss(logits, labels):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
  loss = tf.reduce_mean(cross_entropy)
  return loss

def __main__():

    input_images_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_PIXELS))
    input_deltas_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NB_INSTANTIA_PARAMS - 1))
    output_images_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_PIXELS))

    output = tf.Variable(tf.zeros([IMAGE_PIXELS, IMAGE_PIXELS]))
    for i in range(NB_CAPSULE):
        output += capsule(input_images_placeholder, input_deltas_placeholder)

    loss_op = loss(output, output_images_placeholder)

    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train = optimizer.minimize(loss_op)
    init = tf.global_variables_initializer()

    session = tf.Session()

    session.run(init)

    saver = tf.train.Saver()

    train_set = mnist.dataset()

    for i in tqdm(range(NB_TRAIN_EPOCH)):
        delta = np.zeros([2])
        images_feed, labels_feed = train_set.next_batch(BATCH_SIZE, delta)
        feed_dict = {
            input_images_placeholder: images_feed,
            input_deltas_placeholder: delta,
            output_images_placeholder: labels_feed,
        }
        _, loss_value = session.run([train, loss_op], feed_dict=feed_dict)
        if i % 10 == 0:
            print "loss", loss_value
            saver.save(session, "./model")


if __name__ == "__main__":
    __main__()