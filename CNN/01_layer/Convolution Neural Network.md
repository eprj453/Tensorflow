# Convolution Neural Network

이미지 분류에서 가장 널리 사용된다.

convolution layer, pooling layer, fully-connected-layer 총 3개의 layer로 구성되어 있다.

![A-Convolutional-Neural-Network](https://user-images.githubusercontent.com/52685258/79123750-c97add80-7dd5-11ea-80f9-d285ae6c4c52.png)



![1_8doPbLhwBBafD5NMFqYd4A](https://user-images.githubusercontent.com/52685258/79125181-aaca1600-7dd8-11ea-8e62-f907f79401b3.png)







32x32x3의 이미지에 5x5x3 filter를 적용시킨다. 이를 하나의 점으로 만들어 전체 이미지를 거치며 연산을 진행하면 28x28x1 형태의 feature map(activation map)이 만들어진다. 



![20200413_225218](https://user-images.githubusercontent.com/52685258/79125676-802c8d00-7dd9-11ea-892c-e01135261a1f.png)

각기 다른 6개의 filter를 사용하면서 여러 개의 feature map을 겹치게 되면 6x28x28 형태의 feature map 모음이 생기게 되고, 이것이 새로운 이미지의 크기가 된다. (output layer의 feature 갯수는 convolution filter의 숫자와 같다.)





```python
import numpy as np
import tensorflow as tf
# import tf.keras.layers.Conv2D
from tensorflow import keras
import matplotlib.pyplot as plt

print(tf.__version__)
print(keras.__version__)

tf.enable_eager_execution()

image = tf.constant([[[[1], [2], [3]],
                    [[4], [5], [6]],
                    [[7], [8], [9]]]], dtype=np.float32) # 4차원 layer

print(image.shape)
# batch, height, width, channel
plt.imshow(image.numpy().reshape(3,3), cmap='Greys')
plt.show()
```



![20200413_234006](https://user-images.githubusercontent.com/52685258/79129461-37c49d80-7de0-11ea-8693-c4e7b54581c2.png)