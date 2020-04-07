import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(777)

print(tf.__version__)

x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]

plt.scatter(x_data[0][0],x_data[0][1], c='red' , marker='^')
plt.scatter(x_data[3][0],x_data[3][1], c='red' , marker='^')
plt.scatter(x_data[1][0],x_data[1][1], c='blue' , marker='^')
plt.scatter(x_data[2][0],x_data[2][1], c='blue' , marker='^')

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(len(x_data)) # tensorflow data API


def preprocess_data(features, labels):  # 데이터 전처리 과정
    features = tf.cast(features, tf.float32)
    labels = tf.cast(labels, tf.float32)
    return features, labels

# log_path = "./logs/xor"
# writer = tf.summary.create_file_writer(log_path)

W1 = tf.Variable(tf.random.normal([2, 1]), name='weight1')
b1 = tf.Variable(tf.random.normal([1]), name='bias1')

W2 = tf.Variable(tf.random.normal([2, 1]), name='weight2')
b2 = tf.Variable(tf.random.normal([1]), name='bias2')

W3 = tf.Variable(tf.random.normal([2, 1]), name='weight3')
b3 = tf.Variable(tf.random.normal([1]), name='bias3')


def neural_net(features):
    layer1 = tf.sigmoid(tf.matmul(features, W1) + b1) # layer1
    layer2 = tf.sigmoid(tf.matmul(features, W2) + b2) # layer2
    layer3 = tf.concat([layer1, layer2], -1) # 2개의 layer 병합해서 새로운 layer
    layer3 = tf.reshape(layer3, shape=[-1, 2])
    hypothesis = tf.sigmoid(tf.matmul(layer3, W3) + b3)

    # with writer.as_default():
    #     tf.summary.histogram("weights1", W1, step=step)
    #     tf.summary.histogram("biases1", b1, step=step)
    #     tf.summary.histogram("layer1", layer1, step=step)
    #
    #     tf.summary.histogram("weights2", W2, step=step)
    #     tf.summary.histogram("biases2", b2, step=step)
    #     tf.summary.histogram("layer2", layer2, step=step)
    #
    #     tf.summary.histogram("weights3", W3, step=step)
    #     tf.summary.histogram("biases3", b3, step=step)
    #     tf.summary.histogram("layer3", layer3, step=step)
        #
        # tf.summary.histogram("weights4", W4, step=step)
        # tf.summary.histogram("biases4", b4, step=step)
        # tf.summary.histogram("hypothesis", hypothesis, step=step)

    return hypothesis

def loss_fn(hypothesis, labels):
    cost = -tf.reduce_mean(labels * tf.math.log(hypothesis) + (1 - labels) * tf.math.log(1 - hypothesis))
    return cost

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.01)

def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.float32))
    return accuracy

def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(neural_net(features), labels)
    return tape.gradient(loss_value, [W1, W2, W3, b1, b2, b3])

EPOCHES = 50000

for step in range(EPOCHES+1):
    for features, labels in dataset:
        features, labels = preprocess_data(features, labels)
        grads = grad(neural_net(features), features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W1, W2, W3, b1, b2, b3]))

        if step % 5000 == 0:
            print("Iter : {}, Loss : {:.4f}".format(step, loss_fn(neural_net(features), labels)))

x_data, y_data = preprocess_data(x_data, y_data)
test_acc = accuracy_fn(neural_net(x_data), y_data)

print("Testset Accuracy : {.4f}".format(test_acc))