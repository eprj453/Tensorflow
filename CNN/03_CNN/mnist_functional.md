sequential은 직관적으로 좋지만 몇 가지 문제점이 있다.

- multi input model 불가
- multi output model 불가
- shared-layer(같은 layer 여러번 호출) 불가
- data 흐름이 sequential이 아닌 경우 어려움



위와 같은 문제로 각 layer마다 어떤 layer를 input으로 받고 있는지 명시해주고, 초기 input은 keras.Input()으로 설정해준다.



