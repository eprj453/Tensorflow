import tensorflow as tf
import numpy as np

tf.enable_eager_execution()
tf.set_random_seed(777)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]

# y_data -> one hot encoding
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0],
          [1, 0 ,0]]

x_data = np.asarray(x_data, dtype=np.float32)
y_data = np.asarray(y_data, dtype=np.float32)

nb_classes = 3 # num classes

W = tf.Variable(tf.random.normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random.normal([nb_classes]), name='bias')

variables = [W, b]

# hypothesis = tf.nn.softmax(tf.matmul(x_data, W) + b)

def hypothesis(X):
    return tf.nn.softmax(tf.matmul(X, W) + b)

sample_db = [[8, 2, 1, 4]]
sample_db = np.asarray(sample_db, dtype=np.float32)

# print(hypothesis(sample_db))

def cost_fn(X, Y):
    logits = hypothesis(X)
    cost = -tf.reduce_sum(Y * tf.math.log(logits), axis=1)
    cost_mean = tf.reduce_mean(cost)

    return cost_mean

print(cost_fn(x_data, y_data))

x = tf.constant(3.0)
with tf.GradientTape() as g:
    g.watch(x)
    y = x * x # x^2
dy_dx = g.gradient(y, x) # Will compute to 6.0
print(dy_dx)

def grad_fn(X, Y):
    with tf.GradientTape() as tape:
        cost = cost_fn(X, Y)
        grads = tape.gradient(cost, variables)
        return grads


def fit(X, Y, epoches=2000, verbose=100):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    for i in range(epoches):
        grads = grad_fn(X, Y)
        optimizer.apply_gradients(zip(grads, variables))
        if (i == 0) | ((i+1) % verbose == 0):
            print('Loss at epoches %d : %f' %(i+1, cost_fn(X, Y).numpy()))

fit(x_data, y_data)

a = hypothesis (x_data)
print(a)
print(tf.argmax(a, 1))
