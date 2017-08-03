import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
import copy

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../../MNIST_data')
[sample_image, labels ] = mnist.train.next_batch(200)

print(sample_image.shape)

cnt = 1
tmp = copy.copy(sample_image)
for i in range(200):
    if labels[i] == 9:
        print(i)
        tmp[cnt] = copy.copy(sample_image[i])
        cnt = cnt + 1

tmp = tmp[0:cnt]
print('tmp.shape',tmp.shape, 'cnt', cnt)

sample_image = copy.copy(tmp)
print(sample_image.shape)

sample_image = sample_image[1].reshape([28, 28])
plt.imshow(sample_image, cmap='Greys')