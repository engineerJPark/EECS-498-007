# Lecture 3 : Linear Classifier

Linear Classifier는 basic layer다. 어디에서든 쓰이는 기본이다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled.png)

기본적으로 이미지를 다음과 같이 $32 \times 32 \times 3$  행렬로 생각하고 flatten한 다음, 위 사진과 같이 계산한다. 

$x$가 이미지이고 $W, b$는 각각 weight와 bias이다.

이제 Linear Classifier를 다음 세가지 viewpoint로 볼 것이다.

1. Algebraic Viewpoint
2. Visual Viewpoint
3. Geometric Viewpoint

첫번째로는 Algebraic Viewpoint

여기서는 Linear Classifier가 다음 문제를 푸는 것으로 본다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%201.png)

Weight에 1을 추가하고 bias를 weight에 통합하는 trick이 있으나 잘 쓰이진 않는다

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%202.png)

이미지도 선형성으로 인해 상수곱이 적용되는데, 이런 상황에서(score down)도 일단은 고양이임을 알 수가 있으므로 현실과는 괴리가있다

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%203.png)

이번에는 Visual Viewpoints에서 바라본다.

각 행을 matrix로 변환해서 일종의 이미지로 변환하면 다음과 같은 흐릿한 **template**이 나온다.

그리고 그 template을 학습해서 가장 높은 score를 뽑는 방법을 사용한다고 하자.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%204.png)

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%205.png)

하지만 Linear Classifier는 방향이 변하거나 객체가 여러 개인 경우에 대해서 대응할 수가 없다. 위의 예시의 경우 말 두 마리가 머리를 교차로 두고 겹쳐 있는데, 이런 경우 정확히 예상이 안된다는 것이다.

또한 말이 template이지 그냥 색깔 덩어리라서 기하적 의미가 없다. 색깔만 비슷하면 구분을 못한다...

다음은 Geometric Viewpoint이다. 이번에는 여러 점의 위치를 고려한다.

하나의 픽셀에 대해서 Classifier의 score를 매기면 다음과 같이 한 pixel의 value에 따라 각 Class의 Score가 달라질 것이다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%206.png)

Geometric Viewpoint에서는 각각의 class가 score가 같아지는 등고선이 존재하는거 이다. 그리고 그 등고선에 수직한 방향으로 나아갈 수록 그 score가 커진다.

특정 평면을 넘어가는 score가 나오면 해당 class로 인정한다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%207.png)

다만 차원이 높아지면 직관적이지 않게 되어 도움이 되지 않는다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%208.png)

## Linear Classfier를 기용하기 어려운 이유

---

아래 경우들 때문에 linear classifier로는 구분이 어렵다.

즉, 파란색과 빨간색이 서로 다른 class로 구성되어있는 경우를 얘기하는 것이다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%209.png)

## 어쨌든

Classifier를 거치면 해당 class가 얼마나 정답에 근접한지 score를 구할 수 있다는 것이다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2010.png)

우리는 $W, b$를 구하는건데, 그럼 어떤 식으로 구하는 건가?

1. Loss function 정의 : good $W, b$를 quantify하는 것을 정의
2. Optimization Algorithm 사용 : Loss function 최소화하는 $W, b$ 찾기

## Loss Function = Objective Function = Cost Function

**how good our current classifier is**

Low loss = good

High loss = bad...

## 다른 수식들

dataset 표기

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2011.png)

하나의 sample에 대한 Loss

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2012.png)

dataset 전체의 average Loss

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2013.png)

## Multiclass SVM Loss

대중적인? Loss인 SVM Loss를 살펴본다.

이 경우 correct class의 score가 다른 class 보다 높아야한다.

그리고 Loss와 correct class의 score는 다음과 같이 반비례 관계이다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2014.png)

1위 score와 2위 score 사이의 점수가 Margin을 넘기는 만큼 차이나면 SVM Loss는 0이된다.

Loss는 다음과 같은 식으로 계산된다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2015.png)

측 맞는 class의 score가 틀린 class의 score + 1보다 크면 Loss가 0이 된다는 것이다.

이 경우 이렇고, 계산은 다음과 같이 한다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2016.png)

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2017.png)

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2018.png)

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2019.png)

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2020.png)

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2021.png)

dataset 전체에 대해서는 평균을 낸다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2022.png)

## Questions

### what happens to the loss if the scores for the car image change a bit?

조금만 바뀌는 것은 영향이 없다. loss는 그대로.

### min/max loss?

0 and Infinity

### random scores → what loss would be expected?

predicted scores will be small value이고, 1위 2위 차이가 크지 않으므로 0과 1 중 최대값을 고르는 문제가 된다. 이 문제가 C - 1만큼 있으므로 답은 C - 1.

training 시작할 때 이게 아니라면, 분명 bug가 있다는 의미이다.

### what would happen if the sum were over all classes?

1이다.

근데 이해 불가

### what if the loss used a mean instead of a sum?

preference of weight matrices는 같다. 즉 scale만 다르다.

### 다른 loss를 사용한다면?

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2023.png)

  loss function이 nonlinear해지고, preference of weight matrices도 non trivial해진다.

### W에서 L = 0이면, 이것이 유일한 해인가?

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2024.png)

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2025.png)

위 그림은 W와 2W가 서로 같은 결과를 내놓는다는 의미이다.

그럼 W와 2W 중에서 무엇을 선택하게 강제해야 하나?

---

## **Regularization**

Regularization을 위해 Loss를 다음과 같이 정의한다. 

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2026.png)

Regularization term을 추가함으로써 세가지 효과를 기대할 수 있다.

1. 단순 오차 이외의 선호도를 추가한다.
    1. training error를 최소화하는 것으로는 해결인 안되는 것에 대해, preference를 구별하게 만든다.
2. 과적합 방지
    1. training data에만 fitting이 잘 되지 않도록 한다. 
3. 최적화 개선
    1. curvature을 더해서 optimization이 더 잘되도록 조정한다.

다음은 다양한 Regularization 함수의 종류

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2027.png)

다음과 같이 L2 Regularization term을 넣는다고 하자. 그러면 W2처럼 spread 되어있는게 Loss가 더 낮을 것이다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2028.png)

L2 사용할 때

- data의 noisy
- 많은 feature가 서로 correlated되어 있는 경우
- like to spread out the weights

L1 사용할 때

- L2와는 반대의 상황을 선호
- 하나의 feature(아마 class를 의미하는데 잘못 말한 거 아닐까 싶다.)에 weight을 몰아넣기를 선호한다.
- **결론 -> training data, Objective에 따라서 사용해야 하는 Loss의 종류가 다르다.**

# **Regularization to Simpler Models**

---

![2022-01-17-11-54-26.png](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/2022-01-17-11-54-26.png)

f2가 f1보다 오차는 크지만 generalized가 잘되어있다.

하얀 원이 새로 얻어진 data라고 하면 f1은 overfitting이 되어있어서 bad prediction을 내놓는다.

이렇게 모든 data에 오차가 적게 조정하는 과정을 regularization이라고 한다.

# **기억해야하는 것들**

---

Linear Classifier는 한계가 있다.

SVM Classifier를 통해 L2와 L1 Loss에 대해서 알아보았다.

regularization은 중요하다. regularization을 통해서 model의 목적을 변경할 수 있다.(=특정 data만 predict하도록 변경할 수도 있다는 뜻이다.)

loss function을 교체함으로써 model의 목적을 변경할 수 있다. 즉, 상황에 따라 다른 loss를 사용한다. 언제 model이 좋다고 판단할지도 loss를 변경함으로써 바꿀 수 있다

# **Cross-Entropy Loss**

---

우선 공식을 보자

score = logit, 오른쪽이 softmax function

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2029.png)

cross entropy equation : MLE(Maximum Likelihood Estimation)(likelihood를 maximize하는 weights를 선택하는 과정)

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2030.png)

그동안의 Linear Transform으로 인한 결과를 score라고만 부를 뿐 딱히 그 의미가 없었는데, 이 경우에는 의미가 생긴다. **그 의미는 바로 probability.**

연산과정은 다음과 같다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2031.png)

이제는 다음과 같이 correct data랑 probabilities와 비교를 한다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2032.png)

여기서 KL divergence를 이용해서 Cross entropy를 정의한다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2033.png)

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2034.png)

Cross Entropy와 KL divergence는 probability distribution의 차이를 비교할 때 사용한다.

### Tips

임의의 값을 넣어서 나오는 Loss보다, 학습 초기 단계에서 그 Loss가 평균보다 너무 높게 나오면 그건 문제가 있다는 것이다.

loss, weight initialization 등과 같은 것에 문제가 있을 것이다.

# **Question**

---

### **min/max possible loss of $L_j$ by Cross Entropy Loss**

0과 infinity이다.

다만 SVM 같은 경우는 다른 class의 값이 압도적으로 높으면 Loss가 0이 되는 것이 가능하지만, Cross Entropy의 경우는 그렇지 않다.

원래 label이 one hot된 경우에만 Loss가 0이 나온다. label과 prediction을 예측하는 경우, softmax의 지수승 때문에 Loss가 완전히 0이 되는 것은 불가능하다.

### **all scores are small random values, what is the loss?**

$**-\log (\frac{1}{number \space of \space class}) = \log (number \space of \space class)**$

일단 이 경우 uniform distribution을 기대할 것이고, softmax를 거치므로 ...

**debugging에도 사용할 수 있는데, 예를들어 class가 10개인 CIFAR 10에서는 $\log (10) = 2.3$이 loss의 시작이어야 문제가 없는 것이다.**

$y_i=0$은 ground truth의 index를 의미한다. 

참고로 ground-truth = 학습 대상 데이터의 실제 값을 의미

### 아래의 경우, what is cross entropy loss / SVM loss ?

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2035.png)

cross entropy는 공식 상으로 0보다 클 수 밖에 없고, SVM은 공식을 보면 세가지 경우 모두 0이라서, 그 결과가 0이 된다.

### what happens to each loss if slightly change the scores of the last datapoint?

SVM은 어차피 다 0으로 될 거라서 의미 없다.

cross entropy의 경우, 이로 인해 그 값이 변하게 된다. cross entropy는 맞는 class는 +infinity로 보내려고 하고, 틀린 class는 -infinity로 보내려고 하기 때문이다. 즉 separation이 끊임없이 일어난다.

### what happens to each loss if i double the score of the correct class from 10 to 20?

cross entropy decrease, SVM 0

# Own Question

---

## MLE : Maximum Likelihood Estimation

최대우도법(Maximum Likelihood Estimation, 이하 MLE)은 모수적인 데이터 밀도 추정 방법으로써 파라미터 $θ=(θ_1,⋯,θ_m)$으로 구성된 어떤 확률밀도함수 $P(x|θ)$에서 관측된 표본 데이터 집합을 $x=(x1,x2,⋯,xn)$이라 할 때, 이 표본들에서 파라미터 $θ=(θ_1,⋯,θ_m)$ 를 추정하는 방법이다.

$`x = \lbrace1,4,5,6,9\rbrace`$ 는 왼쪽의 분포를 따를 가능성이 더 높다.

즉 왼쪽 분포가 X에 대한 likelihood가 더 높다는 의미이다.

여기서는 추출된 분포가 정규분포라고 가정했고, 우리는 분포의 특성 중 평균을 추정하려고 했다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2036.png)

Likelihood란, 데이터가 특정 분포로부터 만들어졌을(generate) 확률을 의미한다.

수치적으로 이 가능도를 계산하기 위해서는 **각 데이터 샘플에서 후보 분포에 대한 높이(즉, likelihood 기여도)를 계산해서 다 곱한 것**을 이용할 수 있을 것이다.

곱해주는 것은 모든 데이터들의 추출이 독립적으로 연달아 일어나는 사건이기 때문이다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2037.png)

distribution(분포)이 θ=(μ, σ) 의 parameter를 가지고 있는 정규분포라고 가정하자. 그러면 한 개의 데이터가 이 정규분포를 따를 확률은 다음과 같이 계산할 수 있을 것이다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2038.png)

모든 데이터들이 독립적(independent)이라고 가정하면

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2039.png)

### log likelihood

데이터 X가 θ의 parameter를 가지는 distribution을 따르려면, 이 likelihood가 최대가 되는 distribution을 찾아야 한다.

미분의 편의를 위해서, log와 -를 취해서 그 값이 최소가 되는 값을 구함으로써 maximum likelihood를 만들어주는 값을 구한다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2040.png)

log likelihood 식을 미분하고, 이 식이 0이 되는 값(극솟값)을 찾는다. 이를 만족하는 $\theta$를 찾으면 likelihood를 최대화 할 수 있다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2041.png)

**likelihood를 최대화하는 parameter 값을 maximum likelihood estimate라고 한다.**

**즉, 이렇게 구한 평균 값과 분산 값의 parameter가 정규분포에 대한 maximum likelihood estimate인 것**

하지만 maximum likelihood는 분산을 실제보다 작게 추정하여 표본에 대하여 overfitting될 수도 있다는 한계점도 지닌다.

### Example

정규분포에 대하여..

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2042.png)

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2043.png)

이렇게 하면 바로 likelihood를 최대화하는 parameter 두 가지를 구할 수 있다.

![Untitled](Lecture%203%20Linear%20Classifier%208ae9dfe1d87340cf9a618baff8f2c094/Untitled%2044.png)

## L1 L2 **Regularization**

---

# Reference

[https://angeloyeo.github.io/2020/07/17/MLE.html](https://angeloyeo.github.io/2020/07/17/MLE.html)

[https://process-mining.tistory.com/93](https://process-mining.tistory.com/93)

[https://light-tree.tistory.com/125](https://light-tree.tistory.com/125)

[https://huidea.tistory.com/154](https://huidea.tistory.com/154)