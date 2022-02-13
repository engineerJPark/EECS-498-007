# Lecture 8 : Convolution Architecture

# 주요 idea

---

computation 줄이기

parameter는 줄이면서

NN을 deep하게 만들기

# AlexNet

---

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled.png)

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%201.png)

## Information of each Layers

---

Conv Layer의 정보는 다음과 같이 구해진다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%202.png)

Pooling Layer의 정보는 다음과 같이 구해진다. 단 learnable parameter는 없다.

보다시피, FLOPS이 Conv Layer보다 압도적으로 적다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%203.png)

Flatten과 Fully Connected Layer의 정보는 다음과 같이 구해진다.

Flatten은 parameter와 FLOPS가 없다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%204.png)

총 정리하면 다음과 같다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%205.png)

각 Layer의 Hyperparameter는 trial and error로 정한다...

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%206.png)

오른쪽 세 열로 그래프를 그려보자.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%207.png)

메모리는 주로 초반 Layer에서 집중적으로 사용한다.

Parameter는 Fully Connected Layer에서 늘어난다.

FLOPS는 Convolution Layer에서 늘어난다. FC Layer에서는 줄어든다.

Pooling Layer를 이용해서 FLOPS를 줄인다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%208.png)

# ZFNet

---

bigger net than AlexNet

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%209.png)

stride가 더 작아서 receptive field가 더 커지게 되었다.

filter 개수가 커짐에 따라 model이 더 커지게 되었다.

...

다만 이 시기까지는 model을 크게 만들 수 있는 원칙이 발견되지 못했다.

# VGGNet

---

더 이상 trial and error하지 않고 원칙을 확고히 하게 되었다.

아래 설명을 보자.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2010.png)

## Reason of ‘All Conv are $3 \times 3$ stride 1 pad 1’

---

이 두 가지가 결국 같은 결과를 낳는다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2011.png)

이유는 다음 그림을 보면된다.

$3 \times 3$ Conv Layer를 두번 사용하는 것은 $5 \times 5$ Conv Layer를 한 번 쓰는 것과 같은 receptive field를 받게 된다.

그런데 parameter와 FLOPS는 더 적다!

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2012.png)

즉, 큰 filter를 사용할 필요가 없다는 것이 결론. 그냥 작은 필터 여러 번 쓰세요 :)

## Reason of ‘All Pool are$2 \times 2$ stride 2’ and ‘after pool, double channel’

---

왼쪽이 처음 stage이고, 오른쪽이 그 다음 stage이다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2013.png)

같은 연산량을 쓰면서, 메모리는 반절로하고, parameter는 4배로 키운다!

## AlexNet vs VGG

---

VGG가 압도적으로 더 큰 Network인 것을 알 수 있다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2014.png)

이 시점부터는 model을 GPU별로 나눈 게 아니라 data batch를 GPU별로 병렬 연산했다고 한다.

# GoogLeNet

---

Efficiency : 이 Net의 경우 parameter와 memory와 computation을 줄이면서 결과 수준은 동일하게 나오는 것을 목표로 했다.

## Stem Network

---

VGG는 Conv Layer에서 많은 연산을 할애했다. 이를 보완하기 위해서 downsample을 크게 감행했다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2015.png)

## Inception Module

---

Net 전체에서 반복되는 구조이다. VGG의 conv conv pool 처럼.

VGG에서는 그냥 3*3 conv로 모든 걸 대체했지만, 여기서는 그냥 모든 convolution을 다해보고 합친다.

따라서 hyperparameter로서 conv kernel size를 고민할 필요가 없어진다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2016.png)

또한 1*1 conv를 이용해서 channel의 개수를 줄인다. 이를 bottleneck이라고 한다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2017.png)

## Global Average Pooling

---

끝부분에 배치.

VGG나 AlexNet은 끝부분에 parameter가 넘쳐났다. 이는 Fully Connected Layer 때문이다.

GoogLeNet에서는 avgerage pooling을 이용해서 parameter를 줄임과 동시에 spatial structure를 제거했다.

average pooling은 spatial size = kernel size로 설정했다.

이렇게 length가 channel 방향인 vector 하나가 탄생한다. 그리고 이걸 FC Layer에 넣는다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2018.png)

## Auxiliary Classifier

---

GoogLeNet은 batch normalization 등장 이전에 만들어졌다. 그래서 10+인 Net을 만들기 어려웠는데, 이를 해결하는 방법이 바로 이 Auxiliary Classifier이다.

이를 통해 gradient가 더 쉽게 전파되게 한다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2019.png)

# RESNet : Residual Network

---

Batch Normalization을 통해서 10개 정도의 Layer를 쌓을 수 있게 되었다.

그런데 여기서 더 쌓으니깐 잘 안되는 것이다! overfitting하기 시작하는 것!

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2020.png)

근데 overfitting하는 건 줄 알았는데 알고 보니 underfitting이었던 것이었다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2021.png)

아래와 같은 이유로 아래의 Hypothesis로 문제를 해결하고자 했다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2022.png)

오른쪽과 같은 residual block을 만들어서, identity function을 잘 학습하도록 만들었다.

예를 들면, conv weight을 모두 0을 주면 block은 identity가 될 것이다.

이로써 shallow net을 따라하는 것도 쉬워지고, gradient flow도 잘 되게 만들었다. 왜냐하면 +연산은 computational graph에서 gradient를 복사해가는 효과가 있기 때문

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2023.png)

## Residual Blocks

---

기본적으로 위의 residual block으로 구성되어있다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2024.png)

### Stem Downsample

---

시작부터 크기를 마구 줄인다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2025.png)

### Global Average Pooling

---

마지막 부분에서 FC Layer 없이 Average Pooling으로 spatial structure를 없애고 계산한다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2026.png)

그 결과 initial width과 number of blocks만 선택하면 된다.

이제야 좀 현실에서 바로 사용할만한 수준이 되었다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2027.png)

## Basic Block

---

보통 사용하는 block을 이야기한다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2028.png)

## Bottleneck Block

---

더 deep NN을 만들려고 도입한 block이다.

FLOPS가 조금 줄어든 것을 알 수 있다.

중요한 것은 computational cost를 거의 그대로 두면서 nonlinearity가 증가하고, 더 많은 layer를 쌓았다는 데에 의의를 둬야한다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2029.png)

### 그 결과...

더 깊은 Net을 만들 수 있다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2030.png)

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2031.png)

## 후속 논문에서

---

ReLU의 배치에 관한 내용이 있다. 오른쪽처럼 ReLU를 residual block 내부에 두라는 뜻이다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2032.png)

아주 약간 정확도가 오른다고 한다. 실제로는 별 쓸모 없다고...

## Improving ResNet : ResNext

---

여러 개의 bottleneck를 만든다.

내부의 dimension이 c와 C로 다르다는 것을 유심히 보라.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2033.png)

Computation은 고정한 채로 C, G, c를 변환할 수 있다.

같은 Computation에서 G를 늘리면 performance가 상승한다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2034.png)

참고로 이는 Grouped Convolution하고 그 formulation이 같다.

# Grouped Convolution

---

채널을 나눠서 계산한 다음에 다시 합쳐서 연산한다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2035.png)

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2036.png)

# Squeeze and Excitation Network

---

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2037.png)

# Densely Connected Network

---

layer끼리 Concatenate하는 구조이다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2038.png)

# MobileNet

---

low computation cost를 지향하는 net이다.

block이 다음과 같다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2039.png)

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2040.png)

# NN 구조의 자동화

---

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2041.png)

주황색이 NN search한 결과인데, 대체로 더 좋은 성능을 보이는 것을 알 수 있다.

![Untitled](Lecture%208%20Convolution%20Architecture%2082293ad2ab6a43e7a8d749564aab9812/Untitled%2042.png)

# My Own Question

---

### Floating Point Operations

---

딱 곱셈 후 덧셈한 횟수를 의미한다. 

multiply와 add는 하나의 floating point operation(하나의 clock cycle)으로 취급된다.

# REFERENCE

---

[FLOPS (FLoating point OPerationS) - 플롭스 (tistory.com)](https://hongl.tistory.com/31)