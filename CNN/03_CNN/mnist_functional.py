import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os

# mnist_sequential과 만들고자 하는 결과는 동일

tf.enable_eager_execution()

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

train_images = train_images.astype(np.float32) / 255. # normalization (0~255 사이의 값을 0~1 사이의 실수로 만듬)
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

# 3. build a neural network model
# sequential을 사용해서 일렬로 layer를 배치해 model을 구성하면 이러한 점이 좋지 않다.
# multi-input model 어려움
# multi-output model 어려움
# shared-layer(같은 layer 여러번 호출) model 어려움
# data 흐름이 sequential하지 않은 경우 어려움

# functional API를 사용하면 가능
def create_model():
    # functional API
    # 1. input layer 선언
    # sequential에서는 이전 layer가 당연히 input layer였지만, functional에서는 layer 뒤에 input layer를 명시해줘야 한다.
    inputs = keras.Input(shape=(28, 28, 1))
    conv1 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME', activation=tf.nn.relu)(inputs)
    pool1 = keras.layers.MaxPool2D(padding='SAME')(conv1)

    conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME', activation=tf.nn.relu)(pool1)
    pool2 = keras.layers.MaxPool2D(padding='SAME')(conv2)

    conv3 = keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu)(pool2)
    pool3 = keras.layers.MaxPool2D(padding='SAME')(conv3)

    pool3_flat = keras.layers.Flatten()(pool3)
    dense4 = keras.layers.Dense(units=256, activation=tf.nn.relu)(pool3_flat)

    drop4 = keras.layers.Dropout(rate=0.4)(dense4)
    logits = keras.layers.Dense(units=10)(drop4)

    return keras.Model(inputs=inputs, outputs=logits)

    # residual block
    # inputs = keras.Input(shape=(28, 28, 256))
    # conv1 = keras.layers.Conv2D(filters=64, kernel_size=1, padding='SAME', activation=keras.layers.ReLU())(inputs)
    # conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME', activation=keras.layers.ReLU())(conv1)
    # conv3 = keras.layers.Conv2D(filters=256, kernel_size=1, padding='SAME') # relu는 뒤에서 적용
    
    # add3 = keras.layers.add([conv3, inputs])
    # relu3 = keras.layers.ReLU()(add3)
    # model = keras.Model(inputs=inputs, outputs=relu3)

    # return model

model = create_model()
model.summary()

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

# 6. optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

def evaluate(model, images, labels):
    logits = model(images, training=False) # dropout 미적용
    # logits의 max값과 labels의 max값이 같은지 확인
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    # correct_prediction의 평균을 accuracy로 계산
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

checkpoint  = tf.train.Checkpoint(cnn=model)

print('Learning Start! it takes several minutes')
for epoch in range(training_epochs):
    avg_loss = 0.
    avg_train_acc = 0.
    avg_test_acc = 0.
    train_step = 0
    test_step = 0

    for images, labels in train_dataset:
        # gradient 계산
        grads = grad(model, images, labels)
        # optimizer에 갱신된 gradient 적용해 weight에 반영 
        optimizer.apply_gradients(zip(grads, model.variables))

        # optional
        loss = loss_fn(model, images, labels)
        acc = evaluate(model, images, labels)
        avg_loss = avg_loss + loss
        avg_train_acc = avg_train_acc + acc
        train_step += 1

    avg_loss = avg_loss / train_step
    avg_train_acc = avg_train_acc / train_step

    for images, labels in test_dataset:
        acc = evaluate(model, images, labels)
        avg_test_acc = avg_test_acc + acc
        test_step += 1
    avg_test_acc = avg_test_acc / test_step

    print('Epoch : {}'.format(epoch+1), 'loss=', '{:.8f}'.format(avg_loss),
        'train accuracy=','{:.4f}'.format(avg_train_acc), 
        'test accuracy=','{:.4f}'.format(avg_test_acc))

    checkpoint.save(file_prefix=checkpoint_prefix)

print('Learning Finish!!')  