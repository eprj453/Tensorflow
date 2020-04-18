import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

tf.enable_eager_execution()
# max pooling, average pooling

image = tf.constant([[[[4], [3]],
                    [[2], [1]]]], dtype=np.float32) 

pool = keras.layers.MaxPool2D(pool_size=(2,2), strides=1, padding='same')(image)

print(pool.shape)
print(pool.numpy())

