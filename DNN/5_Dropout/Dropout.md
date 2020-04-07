# Dropout



## Underfit / Overfit

![1__7OPgojau8hkiPUiHoGK_w](https://user-images.githubusercontent.com/52685258/78707859-27c63d00-794c-11ea-8c03-1ef1a276cb45.png)

너무 못 맞춰도 문제(Underfit)

너무 잘 맞춰도 문제(Overfit)

### Overfit이 문제가 되는 이유

Underfit은 직관적으로 봤을 때 데이터를 잘 못맞추기 때문에 문제가 되는 것을 알 수 있다.

Overfit도 문제가 될 수 있는데, 함수를 Train data에 너무 꽉 맞추게 되면 실제 Test data에 맞지 않는 문제가 생긴다.

적절한 fitting이 보지 못한 데이터도 잘 맞출 수 있다.



Dropout은 이를 위해 정규화를 도와주는 과정 중 하나이다.



## Dropout

![Dropout-neural-network-model-a-is-a-standard-neural-network-b-is-the-same-network](https://user-images.githubusercontent.com/52685258/78708739-8f30bc80-794d-11ea-9e4b-e305357f56e4.png)

Input에서 일부 Node를 끄고 Train을 시킨 뒤, Test data를 확인할 때 모든 Node를 켠다.



## Python Code 

```python
def dropout(rate):
    return tf.keras.layers.Dropout(rate) # rate -> Node(Neuran)을 몇퍼센트 끌 것인가
```

```python
class create_model(tf.keras.Model):
    def __init__(self, label_dim):
        super(create_model, self).__init__()
        weight_init = tf.keras.initializers.glorot_uniform()
        self.model = tf.keras.Sequential()
        self.model.add(flatten())

        for i in range(2):
            self.model.add(dense(256, weight_init))
            self.model.add(relu())
            self.model.add(dropout(rate=0.5)) # 50%의 neuran을 끄고 학습
        self.model.add(dense(label_dim, weight_init))

    def call(self, x, training=None, mask=None):
        x = self.model(x)
        return x
```

```python
def loss_fn(model, images, labels):
    logits = model(images, training=True) # training : Dropout 사용여부
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    return loss

```

