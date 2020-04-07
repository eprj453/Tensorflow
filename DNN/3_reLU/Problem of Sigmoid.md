# Problem of Sigmoid



## 기존 학습방향

Input -> Network -> Output

ground Truth - Output = loss

loss를 다시 Network로 학습시키는 과정이 Backpropagation이며 이 loss를 미분한 것이 Gradient(기울기)가 된다.



## Sigmoid

sigmoid function은 양 극단의에 비해 중간의 기울기가 매우 가파르다.

이것이 적을때는 문제가 되지 않지만 여러 개의 Sigmoid function을 학습한다면 매우 작은 값들이 다중으로 곱해져 Gradient가 소실되는 문제가 발생할 수 있다. 이를 Vanishing Gradient라고 한다.



## Relu

f(x) = max(0, x)

x가 0보다 크다면 x를 return하고 그렇지 않다면(음수라면) 0을 return한다.

간단하면서도 성능 향상에 도움이 되기 때문에 널리 쓰인다.

tf.keras.activations에는 sigmoid, tanh, relu, elu, selu 등의 activation이 있고 tf.keras.layers에는 leaky relu가 있다. leaky relu는 0 이하의 값이 들어올 때 x와 일정값 알파를 곱한 값을 반환한다. 0보다 큰 수가 들어올 경우 relu와 동일하게 x를 반환한다.

 





