# Lecture 10 : Training Neural Network 1

# Overview

---

1. One time setup
    1. Activation Function, Data Preprocessing, Weight Initialization, Regularization
2. Training dynamics
    1. Learning rate schedules, large-batch training, hyperparameter optimization
3. After training
    1. Model ensembles, transfer learning

# Activation Function

---

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled.png)

## Sigmoid

- Probability로서의 의미를 지닌다. 마침 딱 [0,1]이라서, boolean 대신해서 쓰일 수 있다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%201.png)

Sigmoid는 위 그림처럼 세가지 문제가 있다.

1. 중앙에서 먼 값은 gradient가 0에 가깝다 → gradient descent 방해

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%202.png)

1. not zero centered : 모든 gradient가 positive 혹은 negative해진다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%203.png)

이렇게 지그재그로 optimizing하게된다. 이게 차원이 많아지면 더 곤란해진다...

물론 이런 문제는 여러 개의 Minibatch를 합치게 되면 어느정도 해결된다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%204.png)

1. exp() 연산은 그 연산량이 크다.

이는 cpu나 mobile에서는 제일 큰 문제다.

...

## Tanh

exp로 표현된 것을 보면, tanh는 사실상 sigmoid를 평행이동 시킨 것이다.

그래서 여전히 같은 문제점을 공유한다.

물론 zero centered하다는 점은 sigmoid보다는 낫다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%205.png)

## ReLU

연산량이 적어 효율적이다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%206.png)

x<0에서는 gradient가 그냥 0, x>0에서는 gradient가 1이다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%207.png)

모든 원소가 0보다 작으면, gradient가 그 뒤로 영원히 0이 되버려서 dead ReLU 현상이 발생한다.

이 경우는 아래 그림과 같이 data cloud 밖으로 완전히 쫓겨난 것을 의미한다.

이를 해결하기 위해 ReLU에 bias를 조금 주기도 한다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%208.png)

## Leaky ReLU

1. x < 0에서의 gradient가 0이 되는 문제를 해결 → 일단 완전한 0은 안된다.
2. zero centered 해결

x < 0에서의 slope는 hyperparameter이다.

PReLU를 이용하면 이 slope조차 Learnable paramter로 변경할 수 있다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%209.png)

## Exponential Linear Unit (ELU)

미분 불가점이 있는 것도 해결 + 좌측 grad이 0이 되는 문제도 해결

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2010.png)

## Scaled Exponential Linear Unit (ELU)

scaled version of ELU

self normalizing property가 있다. 즉, batch norm이 필요없다는 뜻

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2011.png)

## 결론

sigmoid나 tanh같은 거 쓰지말고, Softmax랑 ReLU친구들만 써라.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2012.png)

### 참고

activation function이 일대일 함수가 아니면(역함수가 존재하지 않으면) information loss가 발생한다.

activation function을 사용하는 이유는 information 추가가 아니라 non-linarity를 추가하기 위함이다.

# Data Preprocessing

---

보통의 data가 분산되어있으면, 이를 origin으로 일정하게 끌고 오는 걸 data preprocessing이라고 한다.

보통 Image는 origin에서 먼 결과가 나온다. 그리고 이 data를 Standardize해서 원점으로 끌고 와야한다

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2013.png)

참고로, Standardize를 이용하면 sigmoid의 non zerocentered 문제(Gradient의 모든 원소가 양수 혹은 음수가 되는 문제)를 해결 가능

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2014.png)

image를 상대로 안쓰는 방법 중, Pca(decorrelation)과 whitening 등이 있다.

이는 correlation을 구하고 rotate해서 corelation을 없앤 후, 마지막으로 이를 mean0 var1으로 만든다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2015.png)

왜 data preprocessing을 했나?

Data cloud가 원점에서 멀면, 가중치가 조금만 바뀌어도 원점에서 경계선이 크게 멀어진다. 이를 해결하기 위해.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2016.png)

그동안 image에 대한 data preprocessing은 다음과 같았다...

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2017.png)

### Question

계산 용이 때문이 아니라, 야생의 data를 그대로 사용하는 방법을 학습하기 위해 하는 것이다. Test set에도 같은 방법 적용해야한다.

또한, Batch norm만으로는 data preprocessing의 문제를 해결 불가하다. 미묘하게 성능이 낮을 것이라고...

# Weight Initialization

---

가중치 바이아스 모두  0이면 아웃풋이 모두 0이고, 곱셈 연산과 같은 경우로 인해, 그라디언트도 0이된다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2018.png)

그래서 가중치를 가우시안으로 초기화 해봤으나, deep한 network에선 안되는 문제가 발생했다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2019.png)

그래프를 그려보니, data가 0으로 수렴하는 문제 발생하더라,

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2020.png)

이를 해결하기 위해 deviation을 늘리면, tanh의 gradient 0 영역에 들어가서 이렇게 된다...

여전히 gradient는 0이고, learning은 없다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2021.png)

그래서 gradient 0 영역에는 안들어가면서 data가 0으로 수렴하지 않는 구간을 찾아야한다...

## Xavier initiailization

deviation 대신에, input dimenstion의 sqrt로 나눈다.

이제야 deep함과 상관없이 좋은 data distribution이 나온다!!!

Conv에서는 D_in = inputs to each neuron = channel * kernel size * kernel size이다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2022.png)

Derivation trick = input/output의 Variance가 같아야한다.

여기서 모든 W와 x는 independent하고, identically distributed하다. 따라서 variance 관계식을 저렇게 쓸 수 있다.

또한 세번 째 줄에서는 W, x가 zero mean이어야한다. gaussian distribution에 batch norm을 거쳤으니 타당하다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2023.png)

Tanh는 원점 대칭이라 위처럼 계산해도 괜찮은데, ReLU는 아니다. 완전히 Non Linear하니까..

그래프를 봐도 결국 0으로 수렴하는 것을 알 수 있다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2024.png)

## Kaiming/MSRA Initialization

standard deviation을 sqrt(2/D_in)으로 한다. ReLU가 절반을 죽이고 있으니까 2를 곱한 것

이 방법을 이용하면 VGG를 RBM방법을 이용하지 않고 Initializing할 수 있다. 

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2025.png)

하지만 이 방법도 Residual Net에서는 잘 먹히지 않는다.

왜냐하면... output = input + output구조이다 보니, input의 variance가 그대로 다시 전달되는 것. 즉 layer를 거칠 수록 variance가 커진다.

이를 해결하기위해서 block 내부의 첫 번째 conv는 MSRA로, 두 번째 conv는 0으로 초기화한다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2026.png)

### Question

어디에 global minima가 있을 줄 모르니, 모든 방향으로 gradient가 잘 발생하도록 초기화하는 것이 목표다!

즉, local minima 혹은 saddle point같은 곳에서 시작하지 않도록 이런 초기화 방법을 도입하는 것이다.

# Regularization = Weight Decay

---

## Regularization Term

그럼에도 발생하는 Overfitting. Regularization이 필요하다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2027.png)

loss function에 regularization term을 넣으면 끝!

**Weight Decay라고도 하는데, weight matrix의 L2 norm을 줄여주기 때문이다.**

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2028.png)

## Dropout

각 층의 neuron을 random하게 zero로 만들어서 동작하지 않게 한다.

forward pass에서만 동작해야한다.

probability는 hyperparameter이고, 보통 0.5를 고른다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2029.png)

간단한 implementation을 보자. binary mask를 만들어서 곱해주는 것을 알 수 있다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2030.png)

왜 이걸 하는가?

1. redundant representation을 제거한다. = prevent the co-adaptation of features
    
    → data의 일부만 학습하게 되어서, Net이 Robust해진다.
    
2. Net의 large ensemble을 효율적으로 train하게 된다.
    
    → Dropout을 통해서 서로 weight를 share하는 sub network를 운영하게 된다는 의미이다.
    

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2031.png)

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2032.png)

문제는 Dropout이 test time에서 그 결과물을 random하게 만든다는 문제..

그래서 이 random함을 average out한다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2033.png)

여기서 X는 image input이고, z는 zero mask이다.

위 식을 계산해서 random variable z를 determistic하게 만들어서 test time에 내놓는 것이다.

하지만 이것이 매우 어려운데... (NN의 aribitary함 + Dropout의 random함 때문)

...

test time에는 다음과 같이 된다..

결과물이 반토막 남 → drop 없이 결과물을 dropout probability로 곱한다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2034.png)

test time에서는 training에서 기대한 것과 같은 결과가 나와야하기에,

모든 neuron이 active해야한다. 그리고 dropout probability로 rescaling한다.

즉, train과 test시에 그 구현이 다르다.

...

이는 예시 코드

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2035.png)

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2036.png)

### Inverted dropout

test time이 아닌 training time에 rescaling하고, test time에는 이를 그대로 사용

이렇게 하면 training time에 더 cost를 쓰고, test time이 줄어든다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2037.png)

### Dropout을 집어넣는 곳!

Fully Connected Layer에 넣는다. 여기가 가장 parameter가 많기 때문이다.

하지만 ResNet과 같은 최근 구조는 Dropout을 사용하지 않는다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2038.png)

## Regularization의 common Pattern

train에서 randomness를 넣고, test에서 그걸 평균을 낸다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2039.png)

Batch Normalization의 경우, 한 batch의 output이 다른 batch에 dependent하게 된다.

per minibatch mean/standard deviation ← 각 학습 iteration마다, random element가 각각의 batch에 어떻게 섞여있는지에 의존한다는 의미이다. 

test time에서는 이것들의 average를 이용한다. (fixed value 이용)

...

최근 모델의 경우, L2나 Batch Normalization이 유일한 Regularization이다.

# Data Augmentation

---

새로운 randomness를 준다.

자르고 늘리고 회전하고 뒤집고... 이런 방식으로 같은 형태면 같은 label로 인식되도록 지킨다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2040.png)

이렇게하면 data가 늘어나는 효과가 있다.

대표적으로 Horizontal Flip

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2041.png)

ResNet의 경우, random size로 resize하고, 2*2*4 crop을 해서 input으로 넣는다.

test time의 경우, 5개로 resize하고, 다섯 개의 crop으로 average를 evaluate한다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2042.png)

color jittering

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2043.png)

경우에 따라서 flipping등 상황에 어울리지 않는 방법이 있을 수 있다.

오른손 왼손 구분 등...

이에 대해선 전문적인 도메인 지식이 필요하다.

...

보통은 다음 방법들로 해결한다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2044.png)

...

여기서부터는 정말 흔치 않은 방법들. 이런 방법도 있다는 것만 알아두기

---

DropConnect

여기서는 activation node가 아니라, weight를 zero로 만든다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2045.png)

Fractional Pooling

pooling region을 random하게 고른다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2046.png)

Stochastic Depth

이 경우 Dropblock이라고 볼 수 있다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2047.png)

Cutout

training할 때 한 부분을 0으로 만들어서 가린다. test 시에는 전체 이미지를 다 쓴다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2048.png)

Mixup

두 개의 이미지를 섞은 것. beta distribution으로 섞었다.

![Untitled](Lecture%2010%20Training%20Neural%20Network%201%20fb3c246848214547993680aef3ee605d/Untitled%2049.png)

# Conclusion

---

Fully Connected Layer → Dropout

보통은 Batch Normalization, Data Augmentation, L2 normalization

작은 dataset에 대해서는 cutout, mixup을 기용

# Own Question

---

## Zero - Centered

**Activation 함수의 output 값이 양수, 음수 모두 가능하느냐**

→ 이 여부에 따라 back propagation 에서 parameter 들의 update 속도가 달라진다.

→ RNN(Recurrent Neural Network, 순환신경망)에서 중요하게 요구되는 성질이다.

zero-centered 하지 않다: 

parameter 들의 gradient 들이 서로 다른 부호를 가질 수 없다.→ 특정 벡터 공간에 한 번에 접근할 수 없기에 update 가 느리다.

zero-centered 하다:

parameter 들의 gradient 들이 서로 같은 부호, 다른 부호 모두를 가질 수 있다.→ 특정 벡터 공간에 한 번에 접근할 수 있기에 update 가 빠르다.

Visualization

activation 함수의 zero-centered 의 여부에 따라 weight 들이 update 되는 path 가 달라진다

→ zigzag path 의 여부

→ parameter 들의 update 속도에 영향을 미친다

# REFERENCE

---

[Activation Function: Zero-Centered 의 의미 ((Deep) Neural Network) (tistory.com)](https://sykflyinginthesky.tistory.com/8)