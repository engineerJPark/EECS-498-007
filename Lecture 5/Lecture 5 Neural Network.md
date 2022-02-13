# Lecture 5 : Neural Network

그동안 살펴본 머신러닝 모델은 Linear Model이다.

![Image](https://i.imgur.com/fhPRxI5.jpg)

이러한 경우 문제가 풀리지 않는 것을 지난 번에 배웠다.

![Image](https://i.imgur.com/b6H1cyA.jpg)

컴퓨터 비전에선 이를 Transform으로 해결하는 경우가 많다.

![Image](https://i.imgur.com/BCnyvzG.jpg)

같은 원리로 컬러를 히스토그램으로 나타낸다. 위치 형태와는 무관히

![Image](https://i.imgur.com/NJ6J18a.jpg)

이번에는 위치형태만을 신경쓰고 edge를 찾는다. 이를 벡터로 나타낸다.

![Image](https://i.imgur.com/yYsOY5W.jpg)

이런 방법은 어떤 quality를 채집할지 결정할 Practictioner가 필요하다. 또한, 어떤 transform을 쓸지도 따로 결정해야한다

## Bag of Words (Data Driven)

https://en.m.wikipedia.org/wiki/Data-driven_programming

수많은 이미지에서 random patch를 얻고 이들을 전부 붙인다. 이를 codebook 혹은 set of visual words 라고 부른다.

![Image](https://i.imgur.com/Bu57KtY.jpg)

이미지별로 공통된 구조/feature가 있다. 이를 학습하는 것이 목적.

그 후, codes of visual words를 encode한다.

그리고 color histogram처럼 feature histogram으로 표현한다.

![Image](https://i.imgur.com/u4eZDlN.jpg)

앞서 언급한 세가지 방법을 모두 합쳐서 feature vector를 얻을 수도 있다.

마지막 하나는 input image와 learned template를 innner product한 것

![Image](https://i.imgur.com/6k969Te.jpg)

2011 imagenet 우승팀의 알고리즘을 보면 feature extraction과 기타 등등을 거쳤다.

![Image](https://i.imgur.com/6vXKmbr.jpg)

딥러닝 이전의 모델 구조는 이랬다.

![Image](https://i.imgur.com/UEXJMjM.jpg)

이때의 구조:

Feature extractor no tune

Only Learnable model

즉, feature extracter는 tuning이 안된다.

하지만 우리는 System의 모든 부분이 학습되게 하고 싶다. 그래서 Neural Network를 쓴다.

![Image](https://i.imgur.com/wn9fu8D.jpg)

# Neural Network

NN에서 matrix는 전의 column vector의 각 element가 다음 column vector의 element에 얼마나 영향을 주느냐를 의미하기도 한다.

![Image](https://i.imgur.com/8erSQdy.jpg)

![Image](https://i.imgur.com/CG1kWxH.jpg)

각 element가 모두 연관성이 있으므로, 이를 Dense, Fully Connected Layer 등으로 부른다.

Multi Layer Perceptron = MLP라고도 한다. 1강에 나온다는데 안봐서 몰?루

W1는 template을 learn한다.

W2는 template가 input image가 얼마나 비슷한가...얼마나 응답을 하나...를 의미.

![Image](https://i.imgur.com/aHZgVeA.jpg)

하지만 template이 명확히 어떻게 되가고 있나 interprete할 수 없다. 의미불명

대략적인 spatial structure가 구분 가능해지고 있다는 것 정도만 알 수있다.

![Image](https://i.imgur.com/7vS8hFy.jpg)


![Image](https://i.imgur.com/rOn6MQc.jpg)

이제는 다른 방향으로 있는 동일 물체 → multiple mode도 서로 구분할 수 있다.

hiddent layer는 template을 recombine해서 multiple mode인 같은 class가 같다고 인식되 게 해준다.

하지만 다른 문제가 있는데...

여전히 비슷해보이는 다른 class의 template이 있다는 것.

![Image](https://i.imgur.com/NXmQs0G.jpg)

이런 것들을 Distributed representation이라고 한다.

first layer의 결과는 사람이 해석하기 어렵기 때문이다.

대신 spatial structure가 존재하고, NN은 이 distributed representation을 이용한다.

Q : 하나의 class에 대해 여러 representation을 배우면 문제가 생기지 않나?

한 번에 많이 학습하고, 쓸모없는 거를 삭제하는 방법론이 있다. 이 lecture에서는 다루지 않는다.

# Deep Neural Networks = DNN

depth = layer의 개수 = weight matrices의 개수

width = units의 개수 = dimension of hidden representation

보통은 DNN에서 모든 Layer의 width를 동일하게 설정한다.

![Image](https://i.imgur.com/4OgVGD3.jpg)

# ReLU와 다른 Activation Function

만약 Activation Function 없이 NN을 구성하면 무슨 일이 발생하나?

그저 두 linear transform의 곱이기 때문에 여전히 linear classifier의 역할을 한다.

따라서 activation function을 이용해서 비선형성을 더한다. **즉, activation function은 매우매우 중요하다.**

![Image](https://i.imgur.com/Q1FIF7E.jpg)

그동안은 보통 sigmoid 썼었다. 그러나 지금은 대부분은 ReLU를 사용한다.

![Image](https://i.imgur.com/LKUNFXD.jpg)

정확한 구현을 위해서는 어떤 Activation Function을 사용하는지 논문을 확인하자.

Perceptron의 구조는 다음과 같다.

![Image](https://i.imgur.com/m5XQv30.jpg)


NN은 Multiple templates를 first layer에서 분해하고, 나머지 layer에서 잘 통합한다. 그래서 사용한다.

하지만 다른 이유도 있다.

## Space Warping

linear Transform h = Wx를 생각해보자. x, h는 2 dimensional하다.

저번처럼 등고선을 주는 dot product between weight and input을 생각하자.

![Image](https://i.imgur.com/sAW1iCe.jpg)

line이 Wx의 결과이다. 

이를 geometric space를 linear way로 deform하는 것으로 볼 수 있다. 

![Image](https://i.imgur.com/DiiI5Ta.jpg)

즉, data cloud를 new representation으로 옮기는 것이라고 볼 수 있다.

하지만 그냥 공간만 linear하게 옮기는 것이기에, linearly separable하지 않다는 문제가 있다.

![Image](https://i.imgur.com/7aaZzA5.jpg)

**즉, linear classifier를 두번 가하는건 = FC를 두번 쌓는 것은 representaion을 향상하지 못한다.**

## 하지만 ReLU를 쓴다면 어떨까?

A는 그대로, B는 h2축의 양의 부분으로, C는 origin으로, D는 h1축의 양의부분으로 모인다. 

B는 green feature에 대해서 0이므로, h1축에 대해서는 그 output이 0으로 나오는 것이다. 다른 것도 마찬가지.

![Image](https://i.imgur.com/5NHNxuI.jpg)

이를 data cloud로 다시 표현해봤다.

![Image](https://i.imgur.com/XWAomHc.jpg)

이제야 linearly separable하다. 그리고 input space를 기준으로 보면 nonlinear한 boundary를 형성하는 것을 알 수 있다.

![Image](https://i.imgur.com/fBMEFPO.jpg)

more hidden layer → more non-linearity

![Image](https://i.imgur.com/fJXQW2V.jpg)

위 그림을 보면, overfitting이 걱정되서 dimension을 줄이는(feature을 줄이는)방식을 취할 수 있다.

사실 그것 보다는 regularization term을 사용하는 것이 좋다.

이를 이용하면 boundary를 더 smooth하게 만들 수 있다. → overfitting을 줄인다.

![Image](https://i.imgur.com/qKgwe45.jpg)

# Universal Approximation

NN은 더 다양한 종류의 함수를 표현할 수 있다. Neural Network를 formalize한 것이다.

**하나의 hidden layer가 있는 NN은 $R^n \to R^m$으로 가는 함수를 arbitary precision으로 근사할 수 있다.** 

여기서 arbitary precision이란..... any continuous function을 추정할 수 있다고 말하고자 넣은 단어이다.

기본적으로 hidden layer는 ReLU를 뒤집고 shift하는 기능을 한다.

![Image](https://i.imgur.com/19NM6iT.jpg)

이를 이용해서 bump function을 만들 수 있다.

![Image](https://i.imgur.com/B4NEVOT.jpg)

4, 8, 12...등의 unit을 여러개의 bump 로 보고, 이를 통해 연속적인 함수를 표현할 수 있다. 즉, Wide한 unit으로 구성해서 모든 함수를 표현할 수 있는 것이다.

![Image](https://i.imgur.com/TmMvgmQ.jpg)

![Image](https://i.imgur.com/X6Gur9A.jpg)

물론 실제로 보면 그냥 ReLU의 조합일 뿐이다,

universal approximation을 통해 어떤 함수든 표현 가능하다는 것을 알 수 있다.

하지만 그 모든 함수가 SGD를 통해 학습 가능한지는 알 수 없으며, 근사하는데 얼마나 많은 데이터가 필요한지도 알 수 없다.(KNN도 universal approximator라는 것을 기억해라!)

# Convexity

![Image](https://i.imgur.com/nky8JmC.jpg)

loss function이 convex function인 경우 다음과 같이 bowl모양이고, 이를 optimize하기가 매우 쉽다.

![Image](https://i.imgur.com/VS0DS3A.jpg)

앞서 본 아래와 같은 Loss Function은 모두 convex function이다.

![Image](https://i.imgur.com/GMO6K2Z.jpg)

아래 사진처럼 general한 loss function은 convex할 수도, 아닐 수도 있고, local minima가 있을 수도 있고, 넘어야하는 최대치가 너무 클 수도 있다.

![Image](https://i.imgur.com/uiBtK6f.jpg)

![Image](https://i.imgur.com/5F5AfBv.jpg)

![Image](https://i.imgur.com/s5FQFvm.jpg)

![Image](https://i.imgur.com/akBep3G.jpg)

즉, NN에서 다루는 대부분의 optimizing은 nonconvex하고, 따라서 convergence가 보장되지 않는다.

하지만 경험적으로 잘 동작하는 것 같아서 그대로 사용한다. 연구가 활발한 분야.