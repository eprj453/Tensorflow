import numpy as np
import tensorflow as tf
# import tf.keras.layers.Conv2D
from tensorflow import keras
import matplotlib.pyplot as plt

print(tf.__version__)
print(keras.__version__)

tf.enable_eager_execution()

image = tf.constant([[[[1], [2], [3]],
                    [[4], [5], [6]],
                    [[7], [8], [9]]]], dtype=np.float32) # 4차원 layer의 image

weight = np.array([[[[1.]], [[1.]]],
                  [[[1.]], [[1.]]]])


# print(image.shape)
# # batch, height, width, channel
# plt.imshow(image.numpy().reshape(3,3), cmap='Greys')
# plt.show()
print(weight.shape) # 2,2,2,1

weight_init = tf.constant_initializer(weight)

conv2d = keras.layers.Conv2D(filters=1, kernel_size=2, padding='VALID', kernel_initializer=weight_init)(image)
print('conv2d.shape : ',conv2d.shape)
print(conv2d.numpy().reshape(2,2))
plt.imshow(conv2d.numpy().reshape(2,2), cmap='gray')
plt.show()

# padding을 VALID가 아닌 SAME으로 준다면 출력 또한 3*3으로 나오게 된다.
# FILTER를 여러개 주고 싶다면 WEIGHT의 모양이 2,2,1,3 형태로 나오도록 하고, Conv2D의 filter도 3으로 지정한다.
multi_weight = np.array([[[[1., 10., -1.]], [[1., 10., -1]]],
                        [[[1., 10., -1]], [[1., 10., -1.]]]])

multi_weight_init = tf.constant_initializer(multi_weight)

multi_conv2d = keras.layers.Conv2D(filters=3, kernel_size=2, padding='SAME', kernel_initializer=multi_weight_init)(image)

feature_maps = np.swapaxes(multi_conv2d, 0, 3) # feature도 3개

for i, feature_map in enumerate(feature_maps):
    print(feature_map.reshape(3,3))
    plt.subplot(1, 3, i+1), plt.imshow(feature_map.reshape(3,3), cmap='gray')
plt.show()