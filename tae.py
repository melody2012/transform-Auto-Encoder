# -*-coding: utf-8-*-

import tensorflow as tf
import mnist
import numpy as np
from tqdm import tqdm

IMAGE_PIXELS = 784
NB_CAPSULE = 30
NB_HIDDEN1_UNITS = 10
NB_HIDDEN2_UNITS = 20
NB_INSTANTIA_PARAMS = 3     # 特征数量+1(包括p)

BATCH_SIZE = 32
TRAIN_EPOCH = 10000
LEARNING_RATE = 1e-4
SAVE_PERIOD = 2000

def dense_layer(_x, nb_last_units, nb_this_units):
    # 产生一个全连接层
    weights = tf.Variable(tf.truncated_normal([nb_last_units, nb_this_units],
                            stddev=1.0 / np.sqrt(float(nb_last_units))))
    biases = tf.Variable(tf.zeros([nb_this_units]))     # 偏置
    layer = tf.nn.relu(tf.matmul(_x, weights) + biases)

    return layer


def capsule(images, delta):
    # 产生一个胶囊
    hidden1 = dense_layer(images, IMAGE_PIXELS, NB_HIDDEN1_UNITS)   # 首先是第一个隐层，连接图片和胶囊

    # 然后是第二个隐层，把图片转化成某种高层次表达（encoder）
    hidden2 = dense_layer(hidden1, NB_HIDDEN1_UNITS, NB_INSTANTIA_PARAMS)

    # x, y, p
    p = [hidden2[:, 2]]     # 这个胶囊的权重
    instantiation_params = hidden2[:, 0:2]  # 高层次表达实例化参数（这里是x，y坐标）
    instantiation_params += delta       # 对其应用某种改变（这里是平移）

    # 重新生成图片（decoder）
    hidden3 = dense_layer(instantiation_params, NB_INSTANTIA_PARAMS - 1, NB_HIDDEN2_UNITS)

    # 利用p重新计算输出图片
    capsule_output_images = tf.matmul(p, dense_layer(hidden3, NB_HIDDEN2_UNITS, IMAGE_PIXELS))

    return capsule_output_images

def loss(predicts, labels):
    # mse loss
    loss = tf.reduce_mean(tf.square(predicts - labels))
    return loss

def __main__():

    print "building model"
    # 输入输出占位符
    input_images_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_PIXELS))
    input_deltas_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NB_INSTANTIA_PARAMS - 1))
    output_images_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_PIXELS))

    # 用来接收累加各个胶囊输出
    output = tf.Variable(tf.zeros([BATCH_SIZE, IMAGE_PIXELS]))
    for i in range(NB_CAPSULE):
        output += capsule(input_images_placeholder, input_deltas_placeholder)

    # loss操作符
    loss_op = loss(output, output_images_placeholder)

    # 批随机梯度下降优化器
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train = optimizer.minimize(loss_op)
    init = tf.global_variables_initializer()

    session = tf.Session()
    session.run(init)   # 初始化

    saver = tf.train.Saver()

    print('Loading data')
    train_set = mnist.dataset()     # mnist数据集

    # 进度条
    progress_bar = tqdm(xrange(TRAIN_EPOCH))

    print "start training"

    for i in progress_bar:
        #delta = np.zeros([BATCH_SIZE, 2])
        delta = np.random.randint(low=-3, high=4, size=(BATCH_SIZE, NB_INSTANTIA_PARAMS - 1))   # 随机位移
        images_feed, labels_feed = train_set.next_batch(BATCH_SIZE, delta)
        feed_dict = {
            input_images_placeholder: images_feed,
            input_deltas_placeholder: delta,
            output_images_placeholder: labels_feed,
        }
        _, loss_value = session.run([train, loss_op], feed_dict=feed_dict)
        progress_bar.set_description("loss = %.5f" % loss_value)
        if i % SAVE_PERIOD == 0:
            saver.save(session, "./model/model")
    print "done"


if __name__ == "__main__":
    __main__()