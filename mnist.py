# -*-coding: utf-8-*-

import gzip
import cPickle
import random
import numpy as np


class dataset:
    def __init__(self):
        with gzip.open('./MNIST-data/mnist.pkl.gz', 'rb') as f:
            (self.image_train, _), (self.image_test, _) = cPickle.load(f)
        self.image_train = self.image_train.reshape(60000, 784)
        self.image_test = self.image_test.reshape(10000, 784)
        self.image_train = self.image_train.astype('float32')
        self.image_test = self.image_test.astype('float32')
        self.image_train /= 255
        self.image_test /= 255

    def next_batch(self, batchSize, delta):
        input_images = self.image_train[random.sample(range(0, 60000), batchSize), :]
        output_image = np.zeros([batchSize, 784])

        for i in range(batchSize):
            for y in range(28):
                for x in range(28):
                    new_x = x + delta[0][0];
                    new_y = y + delta[0][1]
                    if new_x < 28 and new_y < 28 and new_x >= 0 and new_y >= 0:
                        output_image[i][new_x + new_y * 28] = input_images[i][x + y * 28]

        return input_images, input_images

