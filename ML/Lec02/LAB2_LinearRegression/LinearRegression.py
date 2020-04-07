import tensorflow as tf
# tf.enable_eager_execution()


x_data = [1, 2, 3, 4, 5] # 입력
y_data = [2, 4, 6, 8, 10] # 출력

W = tf.Variable(2.9) # 기울기, 아무거나 지정 (랜덤값)
b = tf.Variable(0.5) # y절편, 아무거나 지정 (랜덤값)

learning_rate = 0.01

for i in range(100+1):
    # Gradient descent
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b # 우리의 가정
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    if i % 10 == 0:
        print("{:5} | {:10.4f} | {:10.4} | {:10.6f}".format(i, W.numpy(), b.numpy(), cost))


print(W * 5 + b)


