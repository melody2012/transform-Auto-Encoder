# -*-coding: utf-8-*-

import gzip
import cPickle
import random
import numpy as np


class dataset:
    def __init__(self):
        self.train_counter = 0
        self.test_counter = 0
        with gzip.open('./MNIST-data/mnist.pkl.gz', 'rb') as f:
            (self.image_train, _), (self.image_test, _) = cPickle.load(f)
        self.image_train = self.image_train.reshape(60000, 784)
        self.image_test = self.image_test.reshape(10000, 784)
        self.image_train = self.image_train.astype('float32')
        self.image_test = self.image_test.astype('float32')
        self.image_train /= 255
        self.image_test /= 255

    def imgGenerator(self, batchSz, deltaParamNum, sampleRange=10000):
        sampleRange /= batchSz
        counter = 0
        cached = False
        imageCache = []
        labelCache = []
        deltaCache = []
        while(1):
            if(not cached):
                delta = np.random.randint(low=-2, high=3, size=(batchSz, deltaParamNum))  # 随机位移
                deltaCache.append(delta)
                imageCache.append(self.image_train[counter * batchSz : (counter + 1) * batchSz])
                output_image = np.zeros([batchSz, 784])
                for i in range(batchSz):
                    for y in range(28):
                        for x in range(28):
                            new_x = x + delta[i][0];
                            new_y = y + delta[i][1]
                            if new_x < 28 and new_y < 28 and new_x >= 0 and new_y >= 0:
                                output_image[i][new_x + new_y * 28] = imageCache[counter][i][x + y * 28]

                labelCache.append(output_image)

            yield imageCache[counter], \
                  labelCache[counter], \
                  deltaCache[counter]

            counter +=1
            if(counter >= sampleRange):
                counter = 0
                cached = True