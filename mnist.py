from keras.datasets import mnist
import random
import numpy as np

class dataset:
    def __init__(self):
        print('Loading data')
        (self.image_train, _), (self.image_test, _) = mnist.load_data()
        self.image_train = self.image_train.reshape(60000, 784)
        self.image_test = self.image_test.reshape(10000, 784)
        self.image_train = self.image_train.astype('float32')
        self.image_test = self.image_test.astype('float32')
        self.image_train /= 255
        self.image_test /= 255
        print('Done')

    def next_batch(self, batchSize, delta):
        input_images = self.image_train[random.sample(range(0, 60000), batchSize), :]
        output_image = np.zeros([batchSize, 784])

        for i in range(batchSize):
            for y in range(28):
                for x in range(28):
                    new_x = x + delta[0];
                    new_y = y + delta[1]
                    if new_x < 28 and new_y < 28:
                        output_image[i][new_x + new_y * 28] = input_images[i][x + y * 28]

        return input_images, output_image

    def text_image(self):
        pass

