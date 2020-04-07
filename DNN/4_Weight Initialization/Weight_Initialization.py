# 가중치 초기화

# Network의 목표는 loss가 최소화되는 지점을 찾는 것

# global minimum에 도착하기 전에 local minimum에 빠질 위험도 있고 시작점에 따라 학습효율이 달라질 수 있다.

# valance(분산) = 2 / channel_in + channel_out

# load mnist
import tensorflow as tf
import numpy as np

tf.compat.v1.enable_eager_execution()
def normalize(train_data, test_data):
    train_data = train_data.astype(np.float32) / 255.0
    test_data = test_data.astype(np.float32) / 255.0
    return train_data, test_data

def load_mnist():
    # train 60000장 test 10000장
    mnist = tf.keras.datasets.mnist
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    # mnist에서 반환해주는 데이터의 형식과 tensorflow가 input으로 받는 shape이 달라서
    # expand_dims로 채널을 하나 늘려준다. axis는 채널을 늘릴 위치를 지정한다.
    train_data = np.expand_dims(train_data, axis=-1) # [N, 28, 28] -> [N, 28, 28, 1]
    test_data  = np.expand_dims(test_data, axis=-1)

    train_data, test_data = normalize(train_data, test_data) # 데이터 정규화 (0~255의 값을 0~1로 바꿔줌)
    # print('train_labels before: {}'.format(train_labels.shape))
    # print('test_labels before: {}'.format(test_labels.shape))
    to_categorical = tf.keras.utils.to_categorical
    train_labels = to_categorical(train_labels, 10) # [N, ] -> [N, 10]
    test_labels = to_categorical(test_labels, 10)

def flatten():
    return tf.keras.layers.Flatten()

def dense(channel, weight_init):
    return tf.keras.layers.Dense(units=channel, use_bias=True, kernel_initializer=weight_init)
    # unit -> output 채널을 몇개로 할것인가, use_bias -> bias 사용여부

def relu():
    return tf.keras.layers.Activation(tf.keras.activations.relu)


# reLU 구현때는 model 생성 시 weight를 random으로 지정했지만, 가중치 초기화를 xavior로 하고 싶다면 다음과 같은 방법을 사용한다.

class create_model(tf.keras.Model):
    def __init__(self, label_dim):
        super(create_model, self).__init__()
        weight_init = tf.keras.initializers.glorot_uniform() # xavier initialization
        # weight_init = tf.keras.initializers.he_uniform()  # he initialization
        self.model = tf.keras.Sequential()  # Sequential
        self.model.add(flatten())  # [N, 28, 28, 1] -> [N, 784]

        for i in range(2):
            # [N, 784]  ->  [N, 256] -> [N, 256]
            self.model.add(dense(256, weight_init))
            self.model.add(relu())

        self.model.add(dense(label_dim, weight_init))

    def call(self, x, training=None, mask=None):
        x = self.model(x)
        return x


