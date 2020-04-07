import matplotlib.pyplot as plt
import tensorflow as tf
tf.enable_eager_execution()

print(tf.__version__)
x_train = [[1., 2.], [2., 3.], [3., 1.], [4., 3.], [5., 3.], [6., 2.]]
y_train = [[0.], [0.], [0.], [1.], [1.], [1.]]

x_test = [[5., 2.]]
y_test = [[1.]]

# x_train과 y_train을 batch size만큼 학습하는 dataset 생성
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))

# x_train이 [1, 2] 형태이기 때문에 W는 행렬 연산을 위해 2, 1 형태로 생성
W = tf.Variable(tf.zeros([2, 1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

def logistic_regression(features):
    hypothesis = tf.divide(1., 1. + tf.exp(tf.matmul(features, W) + b)) # linear한 값을 exp메서드로 sigmoid(곡선형) 형태로 가설 작성
    return hypothesis

# def loss_fn(features, labels):
#     hypothesis = logistic_regression(features) # 가설
#     cost = -tf.reduce_mean(labels) * tf.math.log(loss_fn(hypothesis)) + (1 - labels) * tf.math.log(1 - hypothesis)
#     return cost

def loss_fn(hypothesis, features, labels):
    cost = -tf.reduce_mean(labels * tf.math.log(logistic_regression(features)) + (1 - labels) * tf.math.log(1 - hypothesis))
    return cost

# def grad(hypothesis, features, labels):
#     with tf.GradientTape() as tape:
#         cost = loss_fn(hypothesis, labels) # 가설과 실제값 비교
#     return tape.gradient(cost, [W,b]) # 미분을 통해 갱신된 모델 return

def grad(hypothesis, features, labels): # 경사값(기울기) 계산
    with tf.GradientTape() as tape:
        loss_value = loss_fn(logistic_regression(features),features,labels)
    return tape.gradient(loss_value, [W,b])

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

epoches = 2000
for step in range(epoches+1): # 실제 학습
    for features, labels in iter(dataset): #
        grads = grad(logistic_regression(features), features, labels)
        optimizer.apply_gradients(grads_and_vars = zip(grads, [W, b]))

    if step % 100 == 0:
        print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(logistic_regression(features), features, labels)))


# 추론한 값은 0.5를 기준(Sigmoid 그래프 참조)로 0과 1의 값을 리턴합니다.
# Sigmoid 함수를 통해 예측값이 0.5보다 크면 1을 반환하고 0.5보다 작으면 0으로 반환합니다.
def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
    return accuracy

test_acc = accuracy_fn(logistic_regression(x_test), y_test)


