import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
# from tensorflow.python.keras.api._v1.keras.datasets.mnist import mnist

tf.compat.v1.enable_eager_execution()

print(tf.__version__)
def normalize(train_data, test_data):
    train_data = train_data.astype(np.float32) / 255.0
    test_data = test_data.astype(np.float32) / 255.0
    return train_data, test_data

def load_mnist():
    # train 60000장 test 10000장
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()

    # mnist에서 반환해주는 데이터의 형식과 tensorflow가 input으로 받는 shape이 달라서
    # expand_dims로 채널을 하나 늘려준다. axis는 채널을 늘릴 위치를 지정한다.
    train_data = np.expand_dims(train_data, axis=-1) # [N, 28, 28] -> [N, 28, 28, 1]
    test_data  = np.expand_dims(test_data, axis=-1)

    train_data, test_data = normalize(train_data, test_data) # 데이터 정규화 (0~255의 값을 0~1로 바꿔줌)
    # print('train_labels before: {}'.format(train_labels.shape))
    # print('test_labels before: {}'.format(test_labels.shape))
    train_labels = tf.keras.utils.to_categorical(train_labels, 10) # [N, ] -> [N, 10]
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    # print('train_labels : {}'.format(train_labels.shape))
    # print('test_labels : {}'.format(test_labels.shape))
    # 여기서 to_categorical 함수를 이용해 one hot encoding을 진행한다
    # 숫자 7을 인식하면 one hot encoding 이전에는 숫자 7로 기록이 되는데,
    # 이를 [0, 0, 0, 0, 0, 0, 0, 1, 0 ,0]의 형태로 기록함으로써 loss함수의 효율성을 높힌다.
    return train_data, train_labels, test_data, test_labels
# craete network
def flatten():
    return tf.keras.layers.Flatten()

def dense(channel, weight_init):
    return tf.keras.layers.Dense(units=channel, use_bias=True, kernel_initializer=weight_init)
    # unit -> output 채널을 몇개로 할것인가, use_bias -> bias 사용여부

def relu():
    return tf.keras.layers.Activation(tf.keras.activations.relu)

# create model
class create_model(tf.keras.Model):
    def __init__(self, label_dim):
        super(create_model, self).__init__()
        weight_init = tf.keras.initializers.RandomNormal() # W값 랜덤지정

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

def loss_fn(model, images, labels):
    logits = model(images, training=True)
    # print('labels shape : {}'.format(labels.shape))
    # print('logits shape : {}'.format(logits.shape))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    # label = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    # softmax(logits) = [0.1, 0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0]
    # softmax를 취해 모든 합이 1이 되도록 한다.
    # print('loss : ',loss)
    # print(type(loss))
    return loss

def accuracy_fn(model, images, labels):
    # model에 image를 넣어 어떤 이미지인지 판별
    logits = model(images, training=False)

    # logit, labels의 모양은 [batch size, label_dim]의 형태이다.
    # label_dim, 즉 10개의 원소 중에서 가장 큰 수의 index를 반환하게 된다.
    # logits에서 제일 큰 수와 labels에서 가장 큰 수의 index를 equal함수로 같은지 아닌지 비교한다.
    prediction = tf.equal(tf.argmax(logits, -1), tf.argmax(labels, -1))

    # cast 함수로 예상값을 실수로 바꿔준다.
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    # print('accuracy : ',accuracy)
    # print(type(accuracy))
    return accuracy

def grad(model, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)
    return tape.gradient(loss, model.variables)

train_x, train_y, test_x, test_y = load_mnist()
# print*
learning_rate = 0.001
batch_size = 128 # 기존 train_data 6만장, test 1만장. 메모리에 부담이 있으므로 숫자를 batch_size만큼 한다.

training_epoches = 1
training_iterations = len(train_x) // batch_size
label_dim = 10

# shuffle에서 buffer_size는 들어가는 데이터보다 크게
# prefetch는 학습하고 있는 네트워크에 batch_size만큼 학습시킨다. 효율 올라감

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).\
    shuffle(buffer_size=100000).\
    prefetch(buffer_size=batch_size).\
    batch(batch_size).\
    repeat()

test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).\
    shuffle(buffer_size=100000).\
    prefetch(buffer_size=len(test_x)).\
    batch(len(test_x)).\
    repeat()

# dataset iterator
train_iterator = train_dataset.make_one_shot_iterator()
test_iterator = test_dataset.make_one_shot_iterator()

# model
network = create_model(label_dim)

# training
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# eager mode
checkpoint = tf.train.Checkpoint(dnn=network) # network 학습이 끊겼을때, 다시 재학습을 위해 변경이 되었던 weight를 기록
global_step = tf.train.create_global_step() # 각각 weight가 몇번째 iteration인지 알려주는 기능

start_epoch = 0
start_iteration = 0

for epoch in range(start_epoch, training_epoches):
    for idx in range(start_iteration, training_iterations):
        train_input, train_label = train_iterator.get_next()
        grads = grad(network, train_input, train_label)
        optimizer.apply_gradients(grads_and_vars = zip(grads, network.variables), global_step=global_step)

        train_loss = loss_fn(network, train_input, train_label)
        train_accuracy = accuracy_fn(network, train_input, train_label)

        test_input, test_label = test_iterator.get_next()
        test_accuracy = accuracy_fn(network, test_input, test_label)

        tf.contrib.summary.scalar(name='train_loss', tensor=train_loss)
        tf.contrib.summary.scalar(name='train_accuracy', tensor=train_accuracy)
        tf.contrib.summary.scalar(name='test_accuracy', tensor=test_accuracy)

        # print(type(epoch))
        # print(type(idx))
        # print(type(training_iterations))
        # print(type(train_loss))
        # print(type(train_accuracy))
        # print(type(test_accuracy))
        print('Epoch : [%2d] [%5d/%5d], train_loss : %.8f, train_accuracy : %.4f, test_accuracy : %.4f'\
            %(epoch, idx, training_iterations, train_loss, train_accuracy, test_accuracy))

        # counter+=1






# create model (function version)

# def create_model_function(label   _dim):
#     weight_init = tf.keras.initializer.RandomNormal()
#
#     model = tf.keras.Sequential()
#     model.add(flatten())
#
#     for i in range(2):
#         model.add(dense(256, weight_init))
#         model.add(relu())
#
#     model.add(dense(label_dim, weight_init))
#
#     return model






