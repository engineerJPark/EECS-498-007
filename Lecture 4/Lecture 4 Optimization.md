# Lecture 4 : Optimization

General한 경우에는 수식을 작성해서 bottom을 찾는 것이 불가능하므로 iteration이 가능한 방법을 찾아서 적용한다.

![Screenshot_20220118-084822_YouTube.jpg](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Screenshot_20220118-084822_YouTube.jpg)

Gradient = function이 가장 큰 상승을 하는 방향, 절대값은 그런 위치에서의 slope를 의미한다.

![Screenshot_20220118-090029_YouTube.jpg](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Screenshot_20220118-090029_YouTube.jpg)

Backpropagation을 통해서 4단계가 아닌 1단계만으로 gradient를 구한다.

![Screenshot_20220118-182856_YouTube.jpg](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Screenshot_20220118-182856_YouTube.jpg)

Gradient Descent로 조금씩 local steepest로 전진한다.

여기서 우리가 스스로 정해야하는 HyperParameter가 세가지이다.

1. 가중치 초기화 방법, 분포
2. Epoch수
3. Learning Rate

![Screenshot_20220118-223024_YouTube.jpg](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Screenshot_20220118-223024_YouTube.jpg)

GD를 가하면 다음과 같이 정규화가 잘된 데이터는 local min으로 직선으로 간다.

![Screenshot_20220118-223958_YouTube.jpg](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Screenshot_20220118-223958_YouTube.jpg)

정규화가 안된 찌그러진 모양은 아래처럼 optimizing path가 휘는 경향이 있다.

![Screenshot_20220118-224005_YouTube.jpg](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Screenshot_20220118-224005_YouTube.jpg)

다음과 같은 것이 5가지 추가 되었다.

1. Weight Initialization
2. Number of Steps
3. Learning Rate
4. Batch Size
5. Data Sampling

Batch Size는 최대한 클 수록 좋다. 보통은 2의 제곱 수를 따른다.

물론 민감한 hyperparameter는 아니라고 한다.

Data를 sampling은 가져오는 순서나 random함을 의미한다. 가끔 이게 중요한 문제들이 있다고 한다.

![Screenshot_20220119-010859_YouTube.jpg](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Screenshot_20220119-010859_YouTube.jpg)

모든 Sample을 다 도는 것이 아니라, 일부의 Sample을 이용해서 전체의 Gradient를 추정하는 방법이다.

여기서는 Loss Function을 probabilitically 하다고 한다.

아래 식을 X,Y의 joint probability라고 생각한다.

X,Y가 어느 분포에서 추출되었다고하면, 이들의 기대값을 표현할 수 있다.

그 결과 full expectation에 대한 monte carlo estimation이 나온다.

![Screenshot_20220119-011445_YouTube.jpg](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Screenshot_20220119-011445_YouTube.jpg)

위 두식은 Loss Function, 아래는 그 gradient를 구한 것이다.

## PROBLEMS OF SGD

Feature의 범위가 하나는 넓고 하나는 좁다면, gradient path가 아래와 같이 진동한다. 즉, 정확한 gradient descent가 어렵다.

이를 해결하기 위해서 regularization을 한다.

![Screenshot_20220119-094036_YouTube.jpg](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Screenshot_20220119-094036_YouTube.jpg)

또한 Local Minima 혹은 Saddle Point등의 gradient가 0이지만 Global Minima가 아닌 지점에서 학습이 멈출 수 있다.

![Screenshot_20220119-094331_YouTube.jpg](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Screenshot_20220119-094331_YouTube.jpg)

All Sample의 일부인 Batch에서만 그 결과를 가져오기에, gradient가 noisy해질 수 있다. 즉, 추정한 gradient가 실제 gradient와 거리가 멀 수도 있다는 것.

![Screenshot_20220119-095231_YouTube.jpg](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Screenshot_20220119-095231_YouTube.jpg)

## SGD Momentum

그래서 나온 것이 SGD Momentum.

Gradient를 그대로 사용하는 것이 아니라, 속도랑을 만들어서 속도항에 그 gradient를 저

즉, iteration마다 속도가 바뀐다고 가정하는 것이다.

![Screenshot_20220119-100056_YouTube.jpg](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Screenshot_20220119-100056_YouTube.jpg)

Rho를 decay weight혹은 friction이라고 부른다.

아래는 같은 의미 다른 표기일 뿐이다.

![Screenshot_20220119-100919_YouTube.jpg](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Screenshot_20220119-100919_YouTube.jpg)

SGD Momentum을 사용함으롬서 다음과 같은 문제가 해결된다.

![Screenshot_20220119-101241_YouTube.jpg](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Screenshot_20220119-101241_YouTube.jpg)

## Nestorv Momentum SGD

이 알고리즘은 velocity vector로 이동하고 난 뒤의 미래에 있을 gradient를 예측해서 Momentum SGD를 한다. = velocity 유지됐을 때의 gradient를 구한다.

![Screenshot_20220119-103934_YouTube.jpg](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Screenshot_20220119-103934_YouTube.jpg)

그래서 식도 다르다

![Screenshot_20220119-104936_YouTube.jpg](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Screenshot_20220119-104936_YouTube.jpg)

## Adagrad to RMSProp

Gradient 제곱을 누적한다. 그리고 이걸로 gradient를 나눠서 update

![Screenshot_20220120-085038_YouTube.jpg](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Screenshot_20220120-085038_YouTube.jpg)

근데 grad_square가 너무 클 수 있어서 개선하고자 만든게 바로 RMSProp.

Decaying coefficient로 너무 느려지지 않게 한다.즉 grad square를 조금씩 깎는다.

![Screenshot_20220120-085533_YouTube.jpg](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Screenshot_20220120-085533_YouTube.jpg)

실제로 overshooting이 많이 줄어든 것을 볼 수 있다.

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled.png)

## Adam = RMSProp + Momentum

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%201.png)

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%202.png)

위 그림처럼 각각의 경우가 합해져 있다는 것을 알 수 있다.

t = 0에서 무슨 일이 생기나? $\beta_2 = 0.999$(큰편)를 가정하라.

first step에서는 moment2는 여전히 0에 가깝다. 따라서 w를 업데이트하는데 0에 가까운 수로 나누므로, step이 갑자기 커진다!

이 문제 해결하기 위해 Adam은 Bias Correction을 이용한다. → first/second momentum이 0이되도  가능하다.

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%203.png)

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%204.png)

Tip : $\beta_1 = 0.9, \space \beta_2 = 0.999, \space L.R = 0.001, \space 0.00001, \space 0.000001$ 이면 왠만한 모델에 다 잘 맞는다.

다른 알고리즘과 비교해보자.

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%205.png)

물론 고차원에서는 이 그림처럼 안되는 상황이 많다.

아래는 비교표

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%206.png)

## First-Order Optimization

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%207.png)

이런 방식은 사실 다음 second order optimization의 근사적인 방법이라고 생각할 수도 있다.

## Second-Order Optimization

이 방식은 포물선을 추정해서 그 포물선의 최소점으로 보낸다.

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%208.png)

low curvature이 생기면 크게 step한다는 이점이 있다.

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%209.png)

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%2010.png)

이를 이용해서 update step $w^*$를 계산하면

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%2011.png)

하지만 사용해야하는 메모리 용량이 너무 커서 사용하지 않는다.

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%2012.png)

## In practice

Adam을 기본으로 하되, 종종 SGD + Momentum with many tuning을 채용해라. 

full batch, 특히 second order optimizer → L-BFGS with disabled all sources noise

# Own Question

---

## Rejection Sampling

임의의 수식을 가진 확률밀도함수로부터 랜덤한 샘플을 추출하려면 어떻게 해야할까?

$f(x)=0.3exp(−0.2x2)+0.7exp(−0.2(x−10)2)$

이 함수 f(x)는 정확히 말하면 확률밀도함수라고는 할 수 없다. 왜냐하면 −∞부터 ∞까지 이 함수를 적분했을 때의 전체 면적이 1이 아니기 때문이다. 

확률밀도 함수의 수식은 있지만 해당 함수로부터 sample을 추출하기 어려운 경우 sampling 방법이 필요할 수 있다. 이럴 때 사용할 수 있는 것이 rejection sampling이다.

 이 유사 확률 분포를 ‘타겟 분포(target distribution)’라고 이름 붙이고, f(x)로 쓰도록 하자. 

![Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/pic1.png](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/pic1.png)

rejection sampling의 첫 단계는 제안 분포(proposal distribution)를 설정하는 것이다. 앞으로 제안 분포를 g(x)로 쓰도록 하자.

가령, uniform distribution이 생긴게 가장 단순하기 때문에 uniform distribution을 이용해서 제안 분포를 만들 수 있을 것이다.

$x = \lbrace x|-7\leq x \lt 17\rbrace$, $g(x) = 
  \begin{cases} 
                1/24 & \text{if} -7 \leq x \lt 17 \\
                0 & \text{otherwise}
  \end{cases}$ 초기 $g(x)$를 상수배 해서 target distribution을 모두 포함하게 할 수 있다.

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%2013.png)

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%2014.png)

### Sampling

제안분포 $g(x)$에서 샘플 하나($x0$)를 추출

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%2015.png)

타겟 분포 $f(x)$와 상수배를 취한 제안 분포 $Mg(x)$의 likelihood를 비교

$f(x_0)/(Mg(x_0))$로 비교한다. 1보다 크면 $f(x)$이 더 큰 것이다.

$x = \lbrace x|0\leq x \lt 1\rbrace$을 정의역으로 하는 uniform distribution의 샘플 값과 비교할 수 있도록 하자. 비교를 위해 얻은 uniform distribution의 출력값은 0에서 1사이의 값이 랜덤하게 출력된다는 점...

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%2016.png)

$f(x)$의 높이가 $Mg(x)$만큼 높은 곳일 수록 accept될 확률이 높다는 것을 알 수 있다.

### Result

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%2017.png)

최종적으로 얻어진 샘플들을 원래의 구하고자 했던 타겟 분포 $f(x)$와 함께 히스토그램으로 그리면..

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%2018.png)

## Markov Chain Monte Carlo Estimation, MCMC

마르코프 연쇄의 구성에 기반한 확률 분포로부터 원하는 분포의 정적 분포를 갖는 표본을 추출하는 알고리즘의 한 분류이다. 즉, MCMC는 샘플링 방법 중 하나.

### Monte Carlo Estimation

Monte Carlo는 쉽게 말해 통계적인 수치를 얻기 위해 수행하는 ‘시뮬레이션’ 같은 것. 유한한 시도만으로 정답을 추정하자는 데 의미가 있다.

Monte Carlo 방식의 시뮬레이션 중 가장 유명한 것 중 하나가 원의 넓이를 계산하는 시뮬레이션이다. 아래 영상을 확인하자.

[https://raw.githubusercontent.com/angeloyeo/angeloyeo.github.io/master/pics/2020-09-17-MCMC/pic1.mp4](https://raw.githubusercontent.com/angeloyeo/angeloyeo.github.io/master/pics/2020-09-17-MCMC/pic1.mp4)

MCMC에서는 “통계적인 특성을 이용해 무수히 뭔가를 많이 시도해본다”는 의미에서 Monte Carlo라는 이름을 붙였다고 보면 좋을 것 같다.

### Markov Chain

Markov Chain은 어떤 상태에서 다른 상태로 넘어갈 때, 바로 전 단계의 상태에만 영향을 받는 확률 과정을 의미한다. 강화학습에서도 나온다.

전에 것하고만 영향을 받지 전의 전에 것에는 영향이 없다는 소리.

앞서 MCMC는 샘플링 방법 중 하나라고 하였는데, “가장 마지막에 뽑힌 샘플이 다음번 샘플을 추천해준다”는 의미에서 Markov Chain이라는 단어가 들어갔다고 보면 좋을 것 같다.

### 이제 진짜, Markov Chain Monte Carlo = MCMC

다시 말해 MCMC를 수행한다는 것은 첫 샘플을 랜덤하게 선정한 뒤, 첫 샘플에 의해 그 다음번 샘플이 추천되는 방식의 시도를 무수하게 해본다는 의미를 갖고 있다.

Metropolis 알고리즘에 대한 예시를 들자.

**1. random initialization**

$f(x) = 0.3\exp\left(-0.2x^2\right) + 0.7\exp\left(-0.2(x-10)^2\right)$에 대해서 MCMC를 한다고 하자.

그리고 이 분포에서 random하게 추출을 한 번 한다.

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%2019.png)

**2. 제안 분포로부터 다음 포인트를 추천받기**

symmetric한 확률분포(정규분포)를 제안분포로 사용. $g(x)$로 둔다.

$x_0$를 중심으로 정규분포를 하나 그려본다.

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%2020.png)

새로 만든 정규분포에서 랜덤하게 표본을 추출(이를 추천받는다고 한다.)한다.

![Untitled](Lecture%204%20Optimization%208e81b19d6e6e49f2b7780d45ea27e60e/Untitled%2021.png)

왼쪽은 거절, 오른쪽은 수용되었다.

기준은 $\frac{f(x_1)}{f(x_0)}>1$이다.

이제 수용되었으면 다음 샘플을 추천받고, 아니면 패자부활전으로 간다.

**3. 패자부활전**

원래의 샘플의 위치를 $x_0$라고 하고 제안 분포를 통해 새로 제안 받은 샘플의 위치를 $x_1$이라고 하자.

uniform distribution $U_{(0,1)}$에서 부터 추출한 임의의 샘플 $u$에 대해서,

$\frac{f(x_1)}{f(x_0)}>u$이 성립하면 수용한다. 여기서도 거절하면 $x_1$을 $x_0$으로 설정한 뒤 다음 샘플 $x_2$을 추천 받는다.

### Pseudo Code

```python
1. Initialize x0
2. For i=0 to N−1

Sample u ~ U[0,1])
Sample xnew ~ g(xnew|xi))
If u<A(xi,xnew) = min{1,f(xnew)/f(xi)}
	xi+1 = xnew
else
	xi+1 = xi
```

# Reference

[Markov Chain Monte Carlo - 공돌이의 수학정리노트 (angeloyeo.github.io)](https://angeloyeo.github.io/2020/09/17/MCMC.html)