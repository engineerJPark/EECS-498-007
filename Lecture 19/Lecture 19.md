# Lecture 19 : Generative Models Part 1

Generative Model은 그동안 주구장창했던 Supervised Learning에서 벗어난 학습 형태이기 때문에 Supervised와 Unsupervised를 먼저 정의해야한다.

# Supervised Learning

간단히 말하자면 data x와 label y가 있고, x -> y를 실현하는 function을 학습하는 것이 목적인 learning을 의미한다.

![](https://i.imgur.com/iCf51nw.png)

예를 들면 이런 방법론 등이 있다.

![](https://i.imgur.com/4p3idwC.png)

# Unsupervised Learning

![](https://i.imgur.com/Odahye1.png)

아래는 예시이다. 여기서 Feature Learning이 우리가 할 autoencoder의 큰 범주이다.

보통 중간의 layer에서 latent representation을 학습하는 것을 목적으로 한다.

![](https://i.imgur.com/QDlYj3F.png)

기본적으로, Generative model을 이용해서 unsupervised learning을 하고, 이렇게 얻은 feature를 이용해서 supervised learning을 하는 것이 목적이다.

![](https://i.imgur.com/arqQ8C1.png)

# Discriminative vs Generative Model

새로운 두가지 모델 구별 방법이 있다. 이는 output으로 나오는 확률의 형태에 따라서 달라진다.



Discriminative model은 input image x가 전제된 class y를 구한다.

Generative model은 input image x를 통해 probability distribution을 구한다.

Conditional Generative model은 label y와 input image x를 통해 probability distribution을 구한다.



이 때, 이 probability function을 Density Function이라고 하며, 이 함수값이 높다는 것은 x가 더 likely하다는 의미이다. (x가 해당 위치에 존재하는 것이 더 타당하다는 의미)



또한 Density function은 normalized가 가능하다. 즉, 모두 합하면 그 값이 1이 된다는 것인데, 이는 곳 각각의 label의 density끼리 competing이 있다는 의미이기도 하다.

![](https://i.imgur.com/cQurljU.png)



이렇듯 Discriminative model은 각각의 label끼리 경쟁한다. 

![](https://i.imgur.com/C8AQKG5.png)

![](https://i.imgur.com/zPBq6kD.png)

하지만 image끼리의 competition은 없다. 이게 무슨 말이냐면..



이렇게 label은 기존의 label 중 하나라고 하지만 정작 다른 이미지가 들어가 있는 것을 구분하지 못한다는 것이다.

이미 주어진 label인 cat과 dog 사이에서 고민하고 있다...

![](https://i.imgur.com/L49gaNX.png)

![](https://i.imgur.com/p5VcVTd.png)

Generative Model의 경우 모든 이미지가 probability mass를 높이기 위한 competition을 한다.



존재가 가능한 모든 이미지에 대한 distribution을 learning한다.

즉 받은 이미지에 '실제로 존재할 수도 있는 확률'에 해당하는 score를 준다.

따라서, reasonable probability를 각 이미지에 배치하려면, deep image understading이 필요하다.



또한, 하나의 이미지가 다른 이미지들 보다도 더 존재 가능성이 높다고 해야한다는 점도 어렵게 만드는 요인 중 하나이다.

![](https://i.imgur.com/OBv6h6S.png)

중요한 점은 model이 기존 label과 들어맞지 않는, unreasonable input을 reject할 수 있다는 점이다.

density function은 probability가 아니라 likelihood를 내놓는다. 그렇기에 어떻게든 가장 비슷한 image를 그 label이라고 고른다.

![](https://i.imgur.com/on8OqQw.png)

질문:

- Gernerative Model의 성능지표를 보통 Perplexity라고 하는데, 이는 underlying visual structure가 좋다면, unseen test image에 high probability를  내보낸다는 것(training 되지 않았더라도 probability가 높아야한다. 잘 학습되었다면!)을 이용해서 측정한다.



Conditional Generative Model의 경우, 마찬가지로 어떤 카테고리에도 없는 이미지는 reject가 된다. 일반적인 Generative model과의 차이점은 threshold를 정해야한다는 점 정도?



또한 전과는 다르게 label이 주어져서, 해당 label내에서 모든 image끼리 그 확률을 구하게 된다.

![](https://i.imgur.com/Tamp9jz.png)

잠깐 Bayes' Rule을 보자.

각각의 항은 Discriminative  Model에서 오거나, Generative model에서 오고, 또한 prior의 경우, distribution을 상징하며, training set의 label 개수를 세서 구할 수 있다. 보통 Gaussian distribution을 상징한다.

![](https://i.imgur.com/PxD6g5w.png)

다음은 위에서 언급한 것 정리. 각 model로 할 수 있는 것들 이다.

![](https://i.imgur.com/CGqp6ig.png)

이건 Generative model의 분류

Explicit density - Tractable density 중, Autoregressive model은 test time에는 input image가 들어오면 density function을 내놓는 model이다.

Implicit density는 likellihood가 바로 나오지는 않지만 underlying distribution을 sampling할 수는 있다. 즉 density를 sample하지만 value도 얻지 못한다.

여기서 Variational Autoencoder의 경우 approximate density function을 구하고,

Generative Adverserial Network는 function으로부터 바로 sample을 한다.

![](https://i.imgur.com/hMwBX42.png)

참고로 generate sample하면 density function에서 값이 더 큰것을 더 쉽게 generate하는 것을 알 수 있다.

# Explicit Density Estimation

explicit한 density function을 구해보자. 이 function의 input은 input image x와 learnable weight W이다.

maximizing weight W\*는 아래 필기와 같이 구하고 log 처리를 해서 gradient descent가 가능하도록 만든다.



이 때, dataset에 있는 sample이라면 high probability를 내놓고, 아니라면 low probability를 내놓도록 학습 시킨다.

여기서 f()는 Nueral network가 된다.

![](https://i.imgur.com/jW7OLa4.png)

이제 density function을 명시적으로 표현할 것이다. Data x 는 sub data로 구성되어있다

예를들어 subdata는 이미지의 각 픽셀이 될 수 있다.

계산해서 구하면 사전에 있던 사건들이 전제된 조건부확률 혹은 likelihood가 된다는 것을 알 수 있다.



Autogressive model은 RNN을 표현한다.

![](https://i.imgur.com/e5vewEa.png)

# PixelRNN

이제는 픽셀을 각 순서대로 나눠서 그 density function을 구할 것이다. 이때 구조는 RNN과 Autoregressive model이다. train하면 density function을 얻는 것이다.



모든 픽셀에 hidden state를 놓고 싶다. 그리고 새로 값을 구하는 pixel은 바로 위, 옆에 있는 픽셀에 영향을 받는다. RGB에 대해서 score를 구하고, 그에 대해 softmax를 0 to 255로 가한다. 



Image captioning 할 때랑 같은 원리로 학습한다고 한다.



하지만 Pixel RNN은 매우 느리다는 문제가 있다. 전의 hidden state에 의존하니까.

![](https://i.imgur.com/coLzzjt.png)

참고로 픽셀의 연산 순서는 이렇게 된다고 보면 된다.

처음 시작할 때, Start token을 그 주변에 padding을 해서 RNN에 넣은 그 결과를 첫 번째 픽셀로 사용한다.

![](https://i.imgur.com/yKelzXc.png)

# Pixel CNN

대체 방법으로 PixelCNN도 있다.

Masked Convolution으로 주변 pixel에 대한 dependency를 표현한다

이 때 receptive field는 위와 왼쪽으로 제한된다

이렇게 하면 parallel가능하다

하지만 여전히 sampling time에는 느리다

또한, 여전히 test time에서는 픽셀 하나씩 생성해서 느리다

![](https://i.imgur.com/tNdwFxN.png)

뭔가를 학습한 거 같지만 줌해서 보면 제대로 배운게 없다...!

![](https://i.imgur.com/Vkfsu7U.png)

아래는 Auto regressive model의 장점과 단점이다.

1. Explicit하기에, test time에 input image에 대해서 density function이 명확히 나온다. (unseen data에 대해서 high probability mass가 나온다.)

2. Unconditioned이기 때문에 무엇이 generated 될지에 대한 제한이 없다

3. 물론 이 generated model에서는 edge, global structure local structure, diversity등이 잘 표현되어있다.

여기서 논하지 않은 Conditional한 사항을 넣으면 Conditional generative model을 만들 수 있다

![](https://i.imgur.com/GaYBQHY.png)

학습된 해상도와 다른 해상도의 이미지를 생성할 수 있나??

- 이 강의에서 얘기하는 vanilla version은 output length가 정해져 있어서 안된다.

---

# Variational Autoencoders

![](https://i.imgur.com/ganZEoX.png)

VAE는 DENSITY FUNCTION의 값을 직접적으로 구할 수 없다!!!

**하지만 하한선은 구할 수 있다. 그래서 이걸 최대로 만들 것이다.**

# (Regular, non-variational) Autoencoders

Autoencoder는 확률적 모델이 아니다

비지도학습으로 이미지의 피처를 학습하고자 하는 모델이다. 레이블 없이!!

이렇게 학습한 feature는 그 아래의 supervised learning model에서 사용한다. 하지만 이 z는 관측 불가하다. 이미 학습되면 그 결과로 끝이라서

![](https://i.imgur.com/UVgpc9s.png)

X에서 Z를 뽑는 함수는 Convolution Net이다

Decoder는 이 transposed Convolution 등으로 Upsample하는 부분이다. Decoder는 feature를 reconstruct한다.

encoder와 decoder의 구조 : decoder는 보통 encoder를 filp한 구조이다.

![](https://i.imgur.com/y99aq5P.png)

그리고 input과 output으로 loss function을 만들고, 그 결과 identity function을 학습하게 된다

오른쪽 그림은 AutoEncoder의 예시

![](https://i.imgur.com/EyfYZna.png)

identity function을 학습하는 부분이 유용한 것이 아니다.

유용한 부분은 z가 input data x에 비해서 매우 작은 차원을 가지게 된다는 점이다.

결국 autoencoder가 하는 것은 input data를 압축하는 것이다. 데이터를 압축하고, 압축 데이터를 다시 원본으로 복구하는 과정에서 무언가를 학습한다는 것이다. 이를 위해서 encoder와 decoder 사이의 bottleneck이 중요하다.

이 bottleneck에는 size of layer에 제한이 있다. 이를 통해서 input data보다 작은 픽셀의 이미지가 나오도록 유도한다.

하지만 Variational AutoEncoder에서는 이 Constraint에 다른 것을 넣을 것이다. Probabilistic Constraint를 넣을 것이다.

---

그럼 이제 어떻게 compress하고 이를 유용하게 사용할 수 있을까?

완전히 학습된 이후에는 decoder를 삭제한다. (안쓸거니까. Loss를 구할 때에만 사용한다.)

![](https://i.imgur.com/dX6vozh.png)

encoder를 다른 부분의 initial part로 만들고, 다른 data에 train한다. 이를 통해서 사전에 학습한 feature를 transfer learning에 사용한다.

![](https://i.imgur.com/f3lxDfW.png)

하지만 Autoencoder 자체는 실용적이지 못하다.

그리고 확률적이지 않다 -> trained model에서 new image를 sample할 수 없다. 그냥 feature만 prediction이 가능할 뿐. 즉, generating image는 불가능하다.

![](https://i.imgur.com/oaCNqHp.png)

# Variational Autoencoder

이 모델의 특징

1. raw data로부터 latent feature z를 학습

2. model로부터 sample을 통해서 new data를 generate한다. (test time 얘기)

3. 이 때, x는 image이고, z는 latent factor used to generate x로 친다.

4. training data ${x^{(i)}}^{N}_{i=1}$은 latent representation z으로 부터 나온다고 가정한다.

![](https://i.imgur.com/k3KBHz6.png)

여전히 똑같이 x를 predict한다. 

그냥 autoencoder와 다른 점은 probabilistic하다는 점 뿐이다.

x는 단순히 하나의 image가 아니라 probability distribution이다.

![](https://i.imgur.com/4DMrqpP.png)

이 때, prior P(z)는 distribution이라고 본다. 주로 standartd unit diagonal Gaussian distribution으로 본다.

![](C:\Users\jshac\AppData\Roaming\marktext\images\2022-02-28-13-27-28-image.png)

그리고 conditional probability인 $P_\theta(x | z^{(i)})$를 구한다. 이는 Neural Network를 통해서 구해지며, 이 probability distribution을 parametric form으로 구할 것이다. 바로 mean과 std deviation을 이용해서!

![](https://i.imgur.com/0SdEm9G.png)

각 픽셀을 probability distribution으로 본다. 따라서 dimension of Gaussian = number of pixel of image가 된다.

그리고 이 distribution을 mean과 std deviation으로 표현한다.

그리고 이 Gaussian distribution을 통해서 이미지를 만들어낸다. (conditioned on z)

![](https://i.imgur.com/luic4wD.png)

(수학적인 내용)Gaussian distribution을 이용하므로,

- full covariance matrix를 이용한다. dimension은 이미지의 크기이다.

- 그런데 만약 이미지의 크기가 512 \* 512라면, hidden layer가 너무 커질 것이다.

- 따라서 general Gaussian 대신에 diagonal Gaussian distribution을 선택한다.

- 따라서, Covaiance between pixel이 없다.

- 즉, 각각의 generated image pixel들은 서로 conditionally independent하다. (conditioned on z)



어떻게 학습할 것인가?

Maximum Likelihood를 구한다.

각 이미지 x마다 z를 관측할 수 있다면 conditional generative model p(x|z)를 학습시킬 수 있지만, 이건 계산 불가능하다.



이게 됏으면 방금까지한 모든 문제가 바로 해결되었다. 따라서 우리가 현재 구하는 건 시스템이 자동으로 latent vector z를 구하도록 하는 것 뿐이다.

따라서 probability density function of x를 사용한다.

![](https://i.imgur.com/d3YlNnG.png)

각 term은 decoder와 Gaussian prior for z에서 구한다.

![](https://i.imgur.com/Pb0PBML.png)

![](https://i.imgur.com/63KgELs.png)

하지만 z는 high dimension이라서, 적분을 적정시간 내에 해낼 수가 없다.

![](https://i.imgur.com/P6cHbD0.png)



다른 방법이 있다. 바로 Bayes' Rule을 사용하는 것

![](https://i.imgur.com/5oc3sBr.png)

![](https://i.imgur.com/BTVaEE9.png)

![](https://i.imgur.com/7iUQcBr.png)

하지만 문제가 있다. Neural Network가 구하는 것과는 정반대인 $P_\theta(z | x)$는 계산 방법이 없다!

구하려면 integral 해야하는데, 앞서 말했듯 오래 걸린다는 문제가 있다.

![](https://i.imgur.com/WQVPPRg.png)

그리고 encoder를 따로 학습 시킨다. 이를 auxiliary Neural Network라고도 한다.

input image x에서 output으로 z를 내놓는다.

![](https://i.imgur.com/7IwVJen.png)

그렇게 해서 Bayes' Rule을 approximate density를 구한다.

![](https://i.imgur.com/KDySD7Q.png)

왼쪽은 decoder, 오른쪽은 encoder이다.

![](https://i.imgur.com/8qlcqOC.png)

# Mathematical Analysis

수식을 다시보자.

![](https://i.imgur.com/mbwT6zU.png)

이렇게 log를 이용해서 다 분리한다.

![](https://i.imgur.com/Yg2jqQG.png)

그리고 Expectation으로 변환한다. z에 대한 기대값으로

![](https://i.imgur.com/dGkm1WH.png)

이를 이용해서 앞선 식을 변환하면 다음과 같다.

![](https://i.imgur.com/EvDhxh5.png)

이렇게 해서 나온 각 결과를 다음과 같이 term별로 의미를 부여한다.

![](https://i.imgur.com/Vj5DtTq.png)

![](https://i.imgur.com/FAoL2Z5.png)

하지만 이건 못 구한다. 앞서 말했듯.

![](https://i.imgur.com/sAUmPqa.png)

하지만 Lower bound는 준다. KL divergence는 무조건 0보다는 크기 때문

![](https://i.imgur.com/sAUmPqa.png)

그래서 앞선 식을 쓰면 다음과 같이 부등식으로 정리할 수 있다.

![](https://i.imgur.com/1pz0bmx.png)

지금까지 설명한 Variational Autoencoders는 다음과 같이 정리할 수 있다.

**lower bound가 최대가 되도록하는 parameter를 학습시키는 것이 목적인 것이다.**

![](https://i.imgur.com/t1WmU8r.png)
