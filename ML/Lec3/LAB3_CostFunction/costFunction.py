import numpy as np
import tensorflow as tf

# Cost Function in pure Python

# X = np.array([1, 2, 3])
# Y = np.array([1, 2, 3])
#
# def cost_func(W, X, Y):
#     c = 0
#     for i in range(len(X)):
#         c += (W * X[i] - Y[i]) ** 2 # cost function과 동일
#     return c / len(X)
#
# print("   W   |     cost")
# for feed_W in np.linspace(-3, 5, num=15):
#     curr_cost = cost_func(feed_W, X, Y)
#     print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))


# Gradient Descent

tf.random.set_seed(0)

x_data = [1., 2., 3., 4.]
y_data = [1., 2., 3., 4.]

W = tf.Variable(tf.random.normal([1], -100., 100.)) # (차원, start, end)

for step in range(300):
    hypothesis = W * x_data
    cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    alpha = 0.02 # learning rate
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, x_data) - y_data, x_data))
    descent = W - tf.multiply(alpha, gradient)
    W.assign(descent)

    if step % 10 == 0:
        print('{:5} | {:10.4f} | {:10.6f}'.format(step, cost.numpy(), W.numpy()[0]))