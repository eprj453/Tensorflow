import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os

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



# 3. Build a neural network

def create_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu, 
        padding='SAME', input_shape=(28, 28, 1))) # 여러 이유로 맨 처음 layer에는 input_shape를 넣어주는게 좋다.    
    model.add(keras.layers.MaxPool2D(padding='SAME')) # MaxPool2D 기본값 = size(2, 2) , stride 2
    # filter 수 2배씩 증가
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu, 
        padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu, 
        padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))

    # fully conected layers
    model.add(keras.layers.Flatten()) # vector로 펴주는 역할
    model.add(keras.layers.Dense(256, activation=tf.nn.relu))  # output 256
    model.add(keras.layers.Dropout(0.4)) # 갯수를 줄여주기 위한 dropout
    model.add(keras.layers.Dense(10)) # output 10

    return model

model = create_model()
# model.summary()

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