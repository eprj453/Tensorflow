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

    train_data = np.expand_dims(train_data, axis=-1) # [N, 28, 28] -> [N, 28, 28, 1]
    test_data  = np.expand_dims(test_data, axis=-1)

    train_data, test_data = normalize(train_data, test_data)

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

def dropout(rate):
    return tf.keras.layers.Dropout(rate) # rate -> Node(Neuran)을 몇퍼센트 끌 것인가

def batch_norm():
    return tf.keras.layers.BatchNormalizaition()

class create_model(tf.keras.Model):
    def __init__(self, label_dim):
        super(create_model, self).__init__()
        weight_init = tf.keras.initializers.glorot_uniform() # xavier initialization
        # weight_init = tf.keras.initializers.he_uniform()  # he initialization
        self.model = tf.keras.Sequential()  # Sequential
        self.model.add(flatten())  # [N, 28, 28, 1] -> [N, 784]

        for i in range(2):
            # [N, 784]  ->  [N, 256] -> [N, 256]

            # dropout에서는 relu 뒤에 dropout을 배치했다.
            # 일반적인 순서는 layer, normalization, activation 순으로 가장 많이 작성한다.

            self.model.add(dense(256, weight_init))
            self.model.add(batch_norm())
            self.model.add(relu())

        self.model.add(dense(label_dim, weight_init))

    def call(self, x, training=None, mask=None):
        x = self.model(x)
        return x


def loss_fn(model, images, labels):
    logits = model(images, training=True) # training : Dropout 사용여부
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    return loss

def accuracy_fn(model, images, labels):
    logits = model(images, training=False)
    prediction = tf.equal(tf.argmax(logits, -1), tf.argmax(labels, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
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

        print('Epoch : [%2d] [%5d/%5d], train_loss : %.8f, train_accuracy : %.4f, test_accuracy : %.4f'\
            %(epoch, idx, training_iterations, train_loss, train_accuracy, test_accuracy))