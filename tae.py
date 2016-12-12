# -*-coding: utf-8-*-

import tensorflow as tf
import mnist
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.python.ops import gen_math_ops

IMAGE_PIXELS = 784
NB_CAPSULE = 30
NB_HIDDEN1_UNITS = 10
NB_HIDDEN2_UNITS = 20
NB_INSTANTIA_PARAMS = 3     # 特征数量+1(包括p)

BATCH_SIZE = 32
TRAIN_EPOCH = 50
LEARNING_RATE = 1e-2
SAVE_PERIOD = 21
SAMPLE_RANGE = 60000


def dense_layer(_x, nb_last_units, nb_this_units):
    # 产生一个全连接层
    weights = tf.Variable(tf.truncated_normal([nb_last_units, nb_this_units],
                                              stddev=1.0 / np.sqrt(float(nb_last_units))))
    biases = tf.Variable(tf.zeros([nb_this_units]))     # 偏置
    #layer = tf.nn.sigmoid(tf.matmul(_x, weights) + biases)
    # return layer
    return tf.matmul(_x, weights) + biases


def capsule(images, delta):
    # 产生一个胶囊
    hidden1 = tf.nn.sigmoid(
        dense_layer(images, IMAGE_PIXELS, NB_HIDDEN1_UNITS))   # 首先是第一个隐层，连接图片和胶囊

    # 然后是第二个隐层，把图片转化成某种高层次表达（encoder）
    hidden2 = tf.nn.sigmoid(
        dense_layer(hidden1, NB_HIDDEN1_UNITS, NB_INSTANTIA_PARAMS))

    # x, y, p
    # 这个胶囊的权重
    p = tf.reshape(hidden2[:, 2], shape=[BATCH_SIZE, 1])

    instantiation_params = hidden2[:, 0:2]  # 高层次表达实例化参数（这里是x，y坐标）
    instantiation_params += delta       # 对其应用某种改变（这里是平移）

    # 重新生成图片（decoder）
    hidden3 = tf.nn.sigmoid(
        dense_layer(instantiation_params, NB_INSTANTIA_PARAMS - 1, NB_HIDDEN2_UNITS))

    # 利用p重新计算输出图片
    capsule_output_images = gen_math_ops.mul(
        p, dense_layer(hidden3, NB_HIDDEN2_UNITS, IMAGE_PIXELS))

    return capsule_output_images


def loss(predicts, labels):
    # mse loss
    #loss = tf.reduce_mean(tf.square(predicts - labels))
    loss = tf.nn.sigmoid_cross_entropy_with_logits(predicts, labels)
    return loss


def plot(image, imgNum=16, dim=(4, 4), figsize=(10, 10)):
    plt.figure(figsize=figsize)
    image = image.reshape([BATCH_SIZE, 28, 28])
    for i in range(imgNum):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(image[i, :, :], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def __main__():

    print "building model"
    # 输入输出占位符
    input_images_placeholder = tf.placeholder(
        tf.float32, shape=(None, IMAGE_PIXELS))
    input_deltas_placeholder = tf.placeholder(
        tf.float32, shape=(None, NB_INSTANTIA_PARAMS - 1))
    output_images_placeholder = tf.placeholder(
        tf.float32, shape=(None, IMAGE_PIXELS))

    # 用来接收累加各个胶囊输出
    output = tf.Variable(tf.zeros([BATCH_SIZE, IMAGE_PIXELS]))
    for i in range(NB_CAPSULE):
        output += capsule(input_images_placeholder, input_deltas_placeholder)

    # loss操作符
    loss_op = loss(output, output_images_placeholder)

    # 批随机梯度下降优化器
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train_op = optimizer.minimize(loss_op)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())   # 初始化

    saver = tf.train.Saver()

    print('Loading data')
    train_set = mnist.dataset().imgGenerator(
        BATCH_SIZE, NB_INSTANTIA_PARAMS - 1, SAMPLE_RANGE)     # mnist数据集

    for e in range(TRAIN_EPOCH):
        # 进度条
        progress_bar = tqdm(xrange(SAMPLE_RANGE / BATCH_SIZE))

        loss_sum = 0

        for _ in progress_bar:
            images_feed, labels_feed, delta = train_set.next()
            feed_dict = {
                input_images_placeholder: images_feed,
                input_deltas_placeholder: delta,
                output_images_placeholder: labels_feed,
            }
            _, loss_value = session.run(
                [train_op, loss_op], feed_dict=feed_dict)

            progress_bar.set_description(
                "epoch %d, loss = %.4f" % (e, np.mean(loss_value)))
            loss_sum += np.mean(loss_value)

        print "avg loss = %.4f" % (loss_sum / SAMPLE_RANGE * BATCH_SIZE)

        if (e) % SAVE_PERIOD == 0:
            saver.save(session, "./model/model")

    # for test loss
    images_feed, labels_feed, delta = train_set.next()
    feed_dict = {
        input_images_placeholder: images_feed,
        input_deltas_placeholder: delta,
    }
    test_output = session.run(output, feed_dict=feed_dict)
    plot(images_feed)
    plot(test_output)

    ph1 = tf.placeholder(tf.float32, shape=(None, IMAGE_PIXELS))
    ph2 = tf.placeholder(tf.float32, shape=(None, IMAGE_PIXELS))
    loss_op_t = loss(ph1, ph2)

    test_loss = session.run(loss_op_t, feed_dict={
                            ph1: test_output, ph2: images_feed})
    print np.mean(test_loss)


if __name__ == "__main__":
    __main__()
