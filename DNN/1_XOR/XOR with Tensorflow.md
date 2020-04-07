# XOR with Tensorflow

하나의 모델만으로는 XOR 문제를 해결할 수 없기 때문에 2개 이상의 Logistic Regression 모델을 결합해 XOR 문제를 해결한다.

tensorflow version => 1.15

## Hypothesis

```python
def neural_net(features):
    layer1 = tf.sigmoid(tf.matmul(features, W1) + b1) # layer1
    layer2 = tf.sigmoid(tf.matmul(features, W2) + b2) # layer2
    layer3 = tf.concat([layer1, layer2], -1) # 2개의 layer 병합해서 새로운 layer
    layer3 = tf.reshape(layer3, shape=[-1, 2])
    hypothesis = tf.sigmoid(tf.matmul(layer3, W3) + b3)

    return hypothesis
```

layer1, layer2라는 Logistic Regression을 2개 생성한 뒤 병합된 layer3가 최종 가설이 된다.



```python
def loss_fn(hypothesis, labels):
    cost = -tf.reduce_mean(labels * tf.math.log(hypothesis) + (1 - labels) * tf.math.log(1 - hypothesis))
    return cost

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.01)

def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.float32))
    return accuracy
```

loss function, optimizer, accuracy function



```python
def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(neural_net(features), labels)
    return tape.gradient(loss_value, [W1, W2, W3, b1, b2, b3])
```

기울기를 구하는 방법은 GradientTape를 이용해 구한 뒤 gradient 메서드로 미분값을 구하는 기존 방식과 동일하다.



```python
EPOCHES = 50000

for step in range(EPOCHES+1):
    for features, labels in dataset:
        features, labels = preprocess_data(features, labels)
        grads = grad(neural_net(features), features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W1, W2, W3, b1, b2, b3]))

        if step % 5000 == 0:
            print("Iter : {}, Loss : {:.4f}".format(step, loss_fn(neural_net(features), labels)))
```

학습.



Iter : 0, Loss : 0.8487
Iter : 5000, Loss : 0.6847
Iter : 10000, Loss : 0.6610
Iter : 15000, Loss : 0.6154
Iter : 20000, Loss : 0.5722
Iter : 25000, Loss : 0.5433
Iter : 30000, Loss : 0.5211
Iter : 35000, Loss : 0.4911
Iter : 40000, Loss : 0.4416
Iter : 45000, Loss : 0.3313
Iter : 50000, Loss : 0.2006

