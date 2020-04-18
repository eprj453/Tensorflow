import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

tf.enable_eager_execution()

mnist = keras.datasets.mnist
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.

img = train_images[0]
plt.imshow(img, cmap='gray')
plt.show()

img = img.reshape(-1, 28, 28, 1) # 4차원으로 이미지 변경
img = tf.convert_to_tensor(img)  # Tensor 형식으로 변경

weight_init = keras.initializers.RandomNormal(stddev=0.01)
conv2d = keras.layers.Conv2D(filters=5, kernel_size=3, strides=(2, 2), # filter 5개, filter(kernel) size (3, 3), stride (2,2), 
            padding='SAME', kernel_initializer=weight_init)(img)

print(conv2d.shape)

feature_maps = np.swapaxes(conv2d, 0, 3)
for i, feature_map in enumerate(feature_maps):
    plt.subplot(1, 5, i+1), plt.imshow(feature_map.reshape(14, 14),
    cmap='gray')

plt.show()

# Pooling Layer (Max Pooling)
pool = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(conv2d)
print(pool.shape)

feature_maps = np.swapaxes(pool, 0, 3)
for i, feature_map in enumerate(feature_maps):
    plt.subplot(1, 5, i+1), plt.imshow(feature_map.reshape(7, 7), cmap='gray')

plt.show()

# 단순한 형태에서 점점 확연한 모양이 나오게 된다.
# Low level feature -> Middle level feature -> High level feature -> Classification


