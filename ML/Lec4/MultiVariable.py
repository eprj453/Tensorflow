import tensorflow as tf
import numpy as np
# Hypothesis H(x) = Wx + b

# cost(W) = 동일

# Gradient Descent : cost가 최소화되는 W값을 찾아줌.
# 기울기를 미분한 다음 기울기가 최소화되는 값을 찾아준다.

# 변수가 여러개라면
# H(x1, x2, x3) = w1x1 + w2x2 + w3x3
# H(x1, x2, x3 + .... + xn) = w1x1 + w2x2 + w3x3 + ...... wnxn

x1 = [73., 93., 89., 96., 73.]
x2 = [80., 88., 91., 98., 66.]
x3 = [75., 93., 90., 100., 70.]
y = [152., 185., 180., 196., 142.] # 예측값

# weights
w1 = tf.Variable(1.)
w2 = tf.Variable(1.)
w3 = tf.Variable(1.)
b = tf.Variable(1.)
# w1 = tf.Variable(tf.random.normal([1]))
# w2 = tf.Variable(tf.random.normal([1]))
# w3 = tf.Variable(tf.random.normal([1]))
# b = tf.Variable(tf.random.normal([1]))

hypothesis = (x1 * w1) + (x2 * w2) + (x3 * w3) + b
learning_rate = 0.000001

for i in range(1001):
    with tf.GradientTape() as tape:
        hypothesis = (x1 * w1) + (x2 * w2) + (x3 * w3) + b
        cost = tf.reduce_mean(tf.square(hypothesis - y))
    w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])

    w1.assign_sub(learning_rate * w1_grad)
    w2.assign_sub(learning_rate * w2_grad)
    w3.assign_sub(learning_rate * w3_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 50 == 0:
        print("{:5} | {:12.4f}".format(i, cost.numpy()))



# matrix를 이용해 data 표현
data = np.array([
    [73., 89., 75., 152.],
    [93., 79., 84., 166.],
    [87., 100., 91., 190.],
    [78., 91., 90., 178.],
    [65., 100., 98., 141.]
], dtype=np.float32)

# [ : , :] [행, 렬] 아무것도 써있지 않는 경우 처음부터 끝까지
x_data = data[:, :-1]
y_data = data[:, [-1]]

W = tf.Variable(tf.random.normal([3, 1]))
b = tf.Variable(tf.random.normal([1]))

def predict(x_data):
    return tf.matmul(x_data, W) + b

learning_rate = 0.00001
n_epochs = 2000

for j in range(n_epochs+1):
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean((tf.square(predict(x_data) - y)))

    W_grad, b_grad = tape.gradient(cost, [W, b])

    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if j % 100 == 0:
        print("{:5} | {:10.4f}".format(j, cost.numpy()))