import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os

tf.enable_eager_execution()

# 뉴럴 네트워크 구축과 모델 퍼포먼스 정의에서 조금 다름.

# 1. set hyper parameter
learning_rate = 0.001
training_epochs = 100
batch_size = 100

# option. create a checkpoint
cur_dir = os.getcwd()
ckpt_dir_name = 'checkpoints'
model_dir_name = 'mnist_cnn_seq'

checkpoint_dir = os.path.join(cur_dir, ckpt_dir_name, model_dir_name)
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_prefix = os.path.join(checkpoint_dir, model_dir_name)

# 2. make a pipelining
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# train_images (60000, 28, 28) train_labels (60000,)
# test_images (10000, 28, 28) test_labels (10000,)

train_images = train_images.astype(np.float32) / 255. # normalization (0~255 사이의 값을 0~1 사이의 실수로)
test_images = test_images.astype(np.float32) / 255.

# mnist data의 형태는 CNN으로 바로 사용할 수 없다.
# 그래서 마지막(axis=-1)에 dimension 추가
# CNN 4차원 -> (batch, height, width, channel)
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# labels one hot encoding
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# dataset 생성
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).\
                shuffle(buffer_size=100000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).\
                batch(batch_size)

# CREATE MODEL
class MNISTModel(tf.keras.Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        self.pool1 = keras.layers.MaxPool2D(padding='SAME')
        self.conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        self.pool2 = keras.layers.MaxPool2D(padding='SAME')
        self.conv3 = keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        self.pool3 = keras.layers.MaxPool2D(padding='SAME')
        
        self.pool3_flat = keras.layers.Flatten()
        self.dense4 = keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.drop4 = keras.layers.Dropout(rate=0.4)
        self.dense5 = keras.layers.Dense(units=10)

    
    def call(self, inputs, training=False):
        net = self.conv1(inputs)
        net = self.pool1(net)
        net = self.conv2(net)
        net = self.pool2(net)
        met = self.conv3(net)
        net = self.pool3(net)
        net = self.pool3_flat(net)
        net = self.dense4(net)
        net = self.drop4(net)
        net = self.dense5(net)
        return net

models = []
num_models = 3
for m in range(num_models):
    models.append(MNISTModel())

# 같은 변수 안에 list로 존재하지만 학습은 각각 시켜주기 때문에 loss function, gradient는 변동 없음
# 4. loss function 정의
def loss_fn(model, images, labels):
    logits = model(images, training=True) # training=True -> dropout 적용
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss

# 5. calculate a gradient
def grad(model, images, labels):
    # backpropagation 방법으로 계속 loss 계산
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)
    # 미분된 값 return
    return tape.gradient(loss, model.variables)

# optimizer
optimizer = optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

def evaluate(models, images, labels):
    prediction = tf.zeros_like(labels)
    for model in models:
        logits = model(images, training=False)
        prediction += logits
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

checkpoints = []
for m in range(num_models):
    checkpoints.append(tf.train.Checkpoint(cnn=models[m]))

for epoch in range(training_epochs):
    avg_loss, avg_train_acc, avg_test_acc = 0., 0., 0.
    train_step, test_step = 0, 0

    for images, labels in train_dataset:
        for model in models:
            grads = grad(model, images, labels)
            optimizer.apply_gradients(zip(grads, model.variables))
            loss = loss_fn(model, images, labels)
            avg_loss += loss / num_models
        acc = evaluate(models, images, labels)
        avg_train_acc = avg_train_acc + acc
        train_step += 1

    avg_loss = avg_loss / train_step
    avg_train_acc = avg_train_acc / train_step

    for images, labels in test_dataset:
        acc = evaluate(models, images, labels)
        avg_test_acc = avg_test_acc + acc
        test_step += 1
    avg_test_acc = avg_test_acc / test_step

    print('Epoch : {}'.format(epoch+1), 'loss=', '{:.8f}'.format(avg_loss),
        'train accuracy=','{:.4f}'.format(avg_train_acc), 
        'test accuracy=','{:.4f}'.format(avg_test_acc))

    for idx, checkpoint in enumerate(checkpoints):
        checkpoint.save(file_prefix=checkpoint_prefix+'-{}'.format(idx))

print('Learning Finish!!')