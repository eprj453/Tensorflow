| x1(quiz1) | x2(quiz2) | x3(midterm1) | y(final) |
| :-------: | :-------: | :----------: | :------: |
|    73     |    80     |      75      |   152    |
|    93     |    88     |      93      |   185    |
|    89     |    91     |      90      |   180    |
|    96     |    98     |     100      |   196    |
|    73     |    66     |      70      |   142    |



여러 개의 시험점수가 있을 때, 기말고사 점수를 예측하는 모델을 만들어보자.



## 가설

![clip_image001](https://user-images.githubusercontent.com/52685258/77572282-0ad44780-6f12-11ea-8243-42767c6168f9.png)

![clip_image001](https://user-images.githubusercontent.com/52685258/77572487-6acaee00-6f12-11ea-9c65-c46522071cfa.png)



x에 들어가는 값이 여러개일 경우 다음과 같은 가설이 만들어진다.

n이 점점 많아진다면 전부 세는 것은 비효율적이므로, matrix를 이용해 다음과 같이 정의할 수 있다.



![clip_image001](https://user-images.githubusercontent.com/52685258/77572785-ef1d7100-6f12-11ea-84e9-1b10153e2331.png)

![clip_image001](https://user-images.githubusercontent.com/52685258/77573093-78cd3e80-6f13-11ea-8a49-2d5d0c265646.png)

![clip_image001](https://user-images.githubusercontent.com/52685258/77573156-94384980-6f13-11ea-8109-9ce047f537bb.png)

X, W는 행렬을 표현하기 위해 대문자로 작성했으며, 행렬의 곱으로 간단하고 x가 n개까지 있을 때 가설을 표현할 수 있다.



## 데이터의 건수 

위의 데이터는 총 5개의 데이터(5 row)이며 인스턴스라고도 한다.

위의 표 데이터를 가설함수로 만들면

![clip_image001](https://user-images.githubusercontent.com/52685258/77574420-4d4b5380-6f15-11ea-965a-083fcee8b731.png)

![clip_image001](https://user-images.githubusercontent.com/52685258/77573156-94384980-6f13-11ea-8109-9ce047f537bb.png)

다음과 같이 표현된다.

x의 행이 아무리 늘어나더라도(인스턴스가 아무리 늘어나도), w의 행 갯수만 변하지 않는다면 이 연산은 깨지지 않는다. 그래서 WX가 아니라 XW의 형태로 표현한다.



출력이 여러개인 경우는?

