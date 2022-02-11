# Lecture 14 : Visualizing and Understanding

[https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture14.pdf](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture14.pdf)

약간 GAN의 느낌이 들어있다.

# What’s going on inside Convolutional Networks?
---

대체 각 Layer가 배우는 건 뭔데요???

![Image](https://i.imgur.com/UGgPKoC.png)

이를 경험적인 방식으로 알려주겠다고 하시는 금발근육남

## First Layer : Visualize Filters

FC Layer에서는 하나의 class당 하나의 template을 학습한다고 말했다.

그리고 마지막 layer에서 template과 class간에 dot product를 해서 score를 얻는다.

**그럼 CNN의 경우는?**

input image와 dot product할 filter를 학습한다.

마찬가지로 FC Layer는 앞서 보여준 filter를 학습한다고 생각한다.

edge와 orientation을 학습한다.

![Image](https://i.imgur.com/fThAv9b.png)

같은 방식으로 filter를 관찰하는 게 higher layer에도 좋을까?

아니더라.. 너무 많다... 직관적이지 않다..

![Image](https://i.imgur.com/2ZPYSA3.png)

# Last Layer
---

![Image](https://i.imgur.com/eaaKsRa.png)

AlexNet의 마지막 FCLayer에서 4096 feature에 뭐가 들었는지 보자.

## Nearest Neighbors

그냥 어떤 이미지들 끼리 비슷하다고 처리되는 지 감을 위해서 실험을 해봤다. 실제로 가동한 알고리즘 아니다!

...

query image → alexnet → 4096 vector

test set도 classifier 거친다. → 4096 vector

이 두 가지 vector 모두 Nearest Neighbors 연산을 해준다.

![Image](https://i.imgur.com/GmRxXkl.png)

AlexNet에서 학습된 4096 dimension의 vector를 Nearest Neighbors에 넣는다.

전부 맞는 class가 꼽혔는데, 서로의 pixel이 많이 다른 것을 알 수 있다.

즉 4096 차원의 vector가 image의 pixel 값을 무시하고 찾는 대상의 특성을 가진다는 것을 알 수 있다.

## Dimensionality Reduction

4096차원의 vector를 2 ~ 3차원으로 낮추는 것

PCA를 이용해서 차원을 줄인다. Spatial structure는 최대한 유지하고, 작은 차원으로 projection한다.

T sne도 비슷한 방법론. NonLinear하게, Spatial structure는 최대한 유지하고, 작은 차원으로 projection한다.

Test set을 t-SNE를 먹여서 2차원으로 만들고 그 지점에 이미지를 넣은 것. 포인트가 디짓이다..

Learned feature vector가 그냥 raw pixel이 아니라, class를 학습하는 것임을 알 수 있다

Weight이 아니라 중간의 convolution activation을 확인해서 input image와 비교하는 방법이 있다.

가장 큰 반응을 보이는 image patch를 고르는 것이다

Row는 각 필터이고, 각 필터마다 최고 반응을 보이는 패치를 모아놓은 것이다.
뭉치는 하나의 Layer이다.

아래는 더 깊은 레이어에서 트레이닝 한 것. 그래서 더 넓은 이미지를 패치로 받는다.

어느 픽셀이 분류에 큰 영향을 미치는지 알아보는 방법도 있다. 일부분을 가리는 것.

이렇게 sailency map을 구한다. 어떤 부분이 분류에 큰 영향을 미치는지 알아낸다. 이를 이용해서 Net이 cheat하는 지 알 수 있다. = 코끼리보고 판단하는 게 아니라 땅바닥 색깔보고 판단한다던가 하는 걸 cheat한다고 하면 된다.

---

backpropagation을 통해서 Saliency map을 찾는 방법도 있다.

gradient of the dog score를 각 input image에 대해서 저장한다.

이를 통해서, 각 pixel의 의미가 '이 pixel을 조금 바꿨을 때 classification score에 얼마나 영향을 줄까'가 된다.

이런 gradient를 generate adversarial net을 만드는 데 사용할 수도 있다고...

이 그림을 통해서 classification의 결과에 가장 큰 영향을 주는 존재가 바로 멍멍이임을 알 수 있다.

![Image](https://i.imgur.com/AKYafNF.png)

이를 이용해서 segmentation(테두리 떼내기)이 가능하다.

![Image](https://i.imgur.com/b0oFAI4.png)

# Intermediate Features via guided backprop
---

이제는 중간에 들어있는 Layer에 대해서 얘기해보자.

방금 전처럼, image 하나를 넣고 backpropagate를 시켜서 중간의 filter에 대해서 Intermediate value를 살펴본다.(final class score가 아니라!) 이 과정으로 up/down이 크게 되는 pixel에 색을 칠한다.

참고로, 이런 saliency map은 training이 완전히 끝난 후에 하는 것이 맞다.

이 결과를 그대로 사용하면 잘못나온다. 그래서 back propagation은 guide가 필요하다.

ReLU에 대해서 0보다 작은 곳은 모두 0으로 만들어버린다. 그리고 0이 된 지점에서는 gradient도 0이다.

또한, backpropagation과정에서 음의 upstream gradient도 0으로 mask out한다.

즉, 아래 그림에서 세번째부터는 임의로 추가한 작업이라는 의미이다.

![Image](https://i.imgur.com/Plc0rMP.png)

이렇게 해서 Convnet을 분석하는 이유는 딱히 잘 모르겠다고 하신다.

![Image](https://i.imgur.com/UiGTucL.png)

아래 그림을 보면, 이 과정을 거쳐서 나온 이미지가 '이미지의 어떤 패치가 뉴런에 영향을 크게 주는지'를 알려준다.

![Image](https://i.imgur.com/XewiP47.png)

# Gradient Ascent
---

이 방법은 이미 학습이 완료된 Net을 기준으로 행한다.

그동안은 기본적으로 test set으로 이런 작업을 한정했다. 그러나 이번에는 모든 이미지를 전제로 한다.

하나의 뉴런의 값이 최대가 되도록하는 이미지를 합성하고 싶다.

![Image](https://i.imgur.com/kKuQH4C.png)

f(I)는 이미지가 뉴런으로 들어가서 나온 값이다.

R(I)는 이미지가 자연스러워 보이게 하는 항이다.

이번에는 Net을 학습하는 것이 아니라 image를 학습해서 trained net의 layer에 대해서 최대로 활성화되는 image를 구하는 것이 목표이다.

1. 일단, zero로 구성된 이미지를 만든다.
2. foward pass 처리해서 score를 구한다.
3. backward pass 처리해서 각 image pixel에 대한 gradient를 구한다.
4. image를 계속 update한다.

![Image](https://i.imgur.com/dvKvNEq.png)

근데 이거 그냥 사실상 GAN 아니냐? 했더니 그거 맞음 ㅇㅇ 해버리는 금발근육남;;;;;;;

그래서 regularizer을 넣어서 constrain과 natural함을 넣어준다.

L2 Norm하면 좋지는 않지만 쉽게 constrain 된다고 한다.

어쨌든 이런 식으로 class score를 가장 높게 만들면서 L2 Norm을 가장 작게 만드는 이미지를 만든다.

아래는 그 에시. 형상을 대략 잘 표현한 것을 알 수 있다.

![Image](https://i.imgur.com/1jAMGeT.png)

하지만 regularizer를 변경하면 더 잘 표현할 수 있다!

![Image](https://i.imgur.com/mptIIrn.png)

그런데 이걸 모든 Layer에 가할 수 있다.

![Image](https://i.imgur.com/DqE6hEa.png)

regularizer를 개선하면 더 좋은 결과를 받을 수 있다. 또한 이를 이용해서 GAN을 할 수 있다.

![Image](https://i.imgur.com/722dj9Q.png)

![Image](https://i.imgur.com/oS1CqWM.png)

금발근육남의 사견으로는 L2와 같은 단순한 Regularizer가 Net이 무엇을 보는 지에 대한 더 정확한 데이터를 내놓는다고 한다.

# Adversarial Examples
---

1.Start from an arbitrary image
2.Pick an arbitrary category
3.Modify the image (via gradient ascent) to maximize the class score
4.Stop when the network is fooled

즉, 겉으로 봐서는 알 수 없게 이미지의 아주 조금만 다르게 변경해서 classification이 아예 다르게 나오도록 할 수 있다는 의미이다.

![Image](https://i.imgur.com/A5r2nxL.png)

# Feature Inversion
---

test set image를 받아서 feature extraction을 한다.

이 feature extraction과 동일한 represiontation을 갖는 새로운 이미지를 생성하기 위해 gradient descent를 한다.

이를 위해 Feature를 invert한다.

![Image](https://i.imgur.com/Tj3NrVb.png)

아래는 이런 방식으로 구한 reconstructing 결과.

앞 부분의 layer보다는 뒷 부분의 layer가 더 많은 정보를 잃은 상태임을 알 수 있다.

![Image](https://i.imgur.com/r6GQfFR.png)

# DeepDream : Amplify Existing Features
---

여기서부터는 그냥 재미용

feature를 net을 통해 추출하고, gradient가 그대로 activation value가 되도록한 상태로 backpropagate한다.

여기서 추출된 feature가 더 강하게 activated되게 하고 싶다.

이것도 결국엔 L2 Norm을 최대가 되도록하는 이미지를 찾는 것과 원래가 같다.

![Image](https://i.imgur.com/p7lkZeD.png)

아래는 코드

![Image](https://i.imgur.com/J80O2Iq.png)

여기서 각 Layer마다 다른 결과가 나오게 된다.

Low level Layer의 경우 edge에 집중된 결과가 나오고, High Level의 경우 조금더 추상적인 이미지가 나오게 된다.

![Image](https://i.imgur.com/Zxuh5rB.jpg)

그래서 deep dream의 윗부분에는 이런 그림이 나온다.

![Image](https://i.imgur.com/tcvLCFE.jpg)

이런 것을 이용하면 image generating을 할 수 있다.

# Texture Synthesis

특정 질감 이미지를 크게 만드는 것이다.

![Image](https://i.imgur.com/4KBQgiW.jpg)

## Nearest Neighbor

Neural Network없이, Nearest Neighbor을 이용한 이런 방법이 있다.

![Image](https://i.imgur.com/vBteYxb.jpg)

# Texture Synthesis with Neural Networks: Gram Matrix

Gradient Ascent를 이용해서 같은 문제를 해결하고자 한다.

![Image](https://i.imgur.com/5sVtuRn.jpg)

픽셀 그대로가 아니라 전반적인 공간적 구조만 그대로 가져오고 싶다. = Texture만 가져오고 싶다.

Gram Matrix  : 구조는 다 버리고 텍스쳐특성만 저장한다

Gram Matrix의 의미 : 어떤 channel feature가 반응을 함께 하나?

![Image](https://i.imgur.com/RXLgO7n.jpg)

구하는 법

1. 이미지를 Net에 넣어서 feature volume을 얻는다.

2. 두 개의 채널 벡터로 만든 후, 서로 외적 후 평균낸다.

3. 이는 곧 unnormalized covariance between feature vectors

4. 그램행렬은 구조적특성을 모두 버렸다


# Neural Texture Synthesis

1. ImageNet을 통해서 CNN을 학습
2. forward pass 각 Layer마다 Gram Matrix를 구한다.
3. random noise로 초기 image를 생성한다.
4. 생성한 이미지를 CNN에 넣고 L2를 계산하고, backpropagation을 행한다. image를 개선한다.
5. 4를 반복

![Image](https://i.imgur.com/Gj3zch9.jpg)

아래 그림을 보면  higher layer에서 더 큰 feature image를 얻고 더 나은 texture를 얻는다.

![Image](https://i.imgur.com/Tynu6Ag.jpg)

# Neural Style Transfer: Feature + Gram Reconstruction

Content와 style image 두개를 받고, 두 이미지에서 Gram Matrix를 꺼낸다. 그리고 이 둘을 비교해가면서 이미지를 생성한다.   

두 개의 다른 이미지에 gram matrix를 이용한다.

이제 두 가지 이미지의 특성을 함께 추출할 수 있다.

새로운 gram matrix를 두 이미지의 gram matrix의 weighted sum으로 계산하는 것이다.

![Image](https://i.imgur.com/xLxdGb7.jpg)

전반적인 구조

![Image](https://i.imgur.com/JAIQTBf.jpg)

이제는 이미지의 gram matrix에 대해서 gradient ascent하면 된다

weight를 content image에 더 두느냐, style image에 더 두느냐에 따라 결과 차이가 있다.

![Image](https://i.imgur.com/eNJvvGy.jpg)

style image의 크기에 따라서 그 결과가 차이가 있다.

![Image](https://i.imgur.com/mppToeI.jpg)

여러 개의 Gram Matrix를 평균 냄으로써 여러 가지 Style을 섞을 수도 있다.

![Image](https://i.imgur.com/wGV8b2c.jpg)

# Problem

문제는 이게 진짜 너무 느리다는 점

애초에 deep한 Net을 사용해서 구하는 것이라 당연한 결과이다.

GPU도 많이 쓴다.

Q : texture 위주로 잡는 알고리즘이라, 실사주의 그림하고는 잘 안맞는다.

# Fast Neural Style Transfer

어쨌든 이런 문제를 해결하기 위한 방법으로 따로 Net을 만드는 방법이 있는데, 이를 금발근육남이 해냈다.

input : content image
output : stylized image

![Image](https://i.imgur.com/rUz8wa0.jpg)

train만 오래걸리고, 쓰는 데는 forward pass만 쓰면된다.

# Recall normalization method

Instance norm is for fast style transfer -> high quality

instance normalization이 바로 이 Fast Nueral Style Transfer에 사용된다.

![Image](https://i.imgur.com/VfRvAey.jpg)

![Image](https://i.imgur.com/S9KIlc9.jpg)

하나의 네트워크에 하나의 스타일만 학습시키는 것에서 여러 개의 스타일을 학습시키는 것으로 발전

![Image](https://i.imgur.com/FNMowhh.jpg)

conditional instance normalization : 기존은 scale, shift parameter(normalization에 사용하는 것들)를 학습한다. 이를 각 스타일마다 다르게 학습하는 것. 그리고 스타일마다 다르게 적용하는 것. conv net은 그대로이다.

여러 가지 스타일을 섞을 수도 있다고 한다.

# Summary : Many methods for understanding CNN representations

Activations: Nearest neighbors, Dimensionality reduction, maximal patches, occlusion

Gradients: Saliency maps, class visualization, fooling images,	feature	inversion

Fun: DeepDream, Style Transfer.