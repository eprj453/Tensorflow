# Weight Initialization

Network의 목표는 loss가 최소화되는 것이다.

그러나 가중치의 편차가 크다면 global minimum에 도착하기 전에 local minimum에 빠져버리거나 시작 지점에 따라 학습효율이 큰 차이가 날 수 있다.



![1_XzO88pAveYEr0OJiCCGIpw](https://user-images.githubusercontent.com/52685258/78705032-d025d280-7947-11ea-9ed6-28324822d4a4.png)

왼쪽처럼 학습이 된다면 아주 좋겠지만 모델의 분포가 중구난방이라면 Local minimum에 바지거나 saddle point를 만나 효율이 낮아질 확률이 높다.

- w를 random Normal Initailization으로 설정한다면 평균은 0, 분산이 1인 w값이 설정



그래서 가중치 초기화를 해주는데, 크게 2가지 방법이 있다.



## Xavier Initialization

우리의 network가 어디서 출발할 것인지를 설정해주게 된다.
$$
Variance = 2 / channel In + channelOut
$$
channelIn -> Input으로 들어가는 channel의 수

channelOut -> Output으로 나오는 channel의 수



## He Initialization

relu에 특화되어 있는 초기화 방식
$$
Variance = 4 / channel In + channelOut
$$




sigmoid를 relu로 바꾸는 것도 효율 향상에 도움이 되지만, weight initailiziation으로도 큰 성능 향상을 기대할 수 있다.

