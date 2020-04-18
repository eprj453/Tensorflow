# NN Implementation Flow in Tensorflow

1. Set hyper parameters

   - learning rate, training epochs, batch size, etc

2. Make a data pipelining

   - dataset load, batch size만큼 데이터 쓴 이후 network에 공급하는 역할
   - tf.data, etc

3. Build a neural network model

   - 모델
   - tf.keras sequential APIs, etc

4. Define a loss function

   - cross entropy, etc

5. Calculate a gradient

   - tf.GradientTape(), etc

6. Select an optimizer

   - Weight update

   - Adam optimizer, etc

7. Define a metric for model's performance

   - 모델 성능 측정
   - accuracy

8. (optional) Make a cehckpoint for saving

   - 매번 학습을 다시 할 필요 없도록

9. Train and Validate a neural network model



## 만들고자 하는 CNN 구조

![20200416_005846](https://user-images.githubusercontent.com/52685258/79359357-76438f00-7f7d-11ea-8548-e700d9036089.png)

### summary

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 32)        320
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 128)         73856
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 128)         0
_________________________________________________________________
flatten (Flatten)            (None, 2048)              0
_________________________________________________________________
dense (Dense)                (None, 256)               524544
_________________________________________________________________
dropout (Dropout)            (None, 256)               0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                2570

Total params: 619,786
Trainable params: 619,786
Non-trainable params: 0

_________________________________________________________________





![20200416_160003](https://user-images.githubusercontent.com/52685258/79424602-5d79be80-7ffb-11ea-8c0b-800a036cebbf.png)