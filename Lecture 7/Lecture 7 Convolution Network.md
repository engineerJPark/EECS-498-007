# Lecture 7 : Convolution Network

Linear Classifier는 spatial structure를 반영하지 못하고 flatten만한다.

Conv를 도입하면 해결!

![Image](https://i.imgur.com/FIOFHwH.jpg)

이제 Layer의 component가 FC Layer & Activation Function → Convolution Layer & Pooling Layer & Normalization으로 바뀌는 것을 목도하게 될 것!

![Image](https://i.imgur.com/TusulpZ.jpg)

그중에서 Convolution Layer는 FC Layer의 역할을 하게 된다.

그동안 FC Layer는 다음과 같이 forward pass 과정을 거쳤다. 그저 dot product했다는 뜻

![Image](https://i.imgur.com/Y87fFz7.jpg)

그러나 우리는 앞으로 아래 같은 3*32*32 이미지에서 spatial structure을 유지하는 방식으로 연산을 거치는 layer를 선언할 것이다.

이미지는 기본적으로 **channel(depth) * width * height** 구조로 되어있다.

참고로 저 크기의 image는 대표적으로 CIFAR10이 있겠다.

![Image](https://i.imgur.com/x09sQG4.jpg)

참고로 다음을 기억하는 것이 좋다.

convolution 연산을 하는 장치?를 filter라고 하는데, 이 filter에 저장되어있는 숫자가 Weight가 된다. 즉, 이 filter의 weight를 학습하는 것이 목적이다.

filter의 channel은 input image의 channel과 같다.

filter는 input image를 slide하면서 dot product를 계산해나간다. 

![Image](https://i.imgur.com/lKZ7nr0.jpg)

이 dot product의 의미는 filter와 input image가 그 지점에서 얼마나 match-up하는 지를 의미한다고 볼 수 있다.

참고로 당연히 bias도 있는데 자주 생략하곤 한다.

![Image](https://i.imgur.com/LQYH3lB.jpg)

이렇게 Convolution 연산을 가하고 나온 output을 feature map 혹은 activation map이라고 한다.

![Image](https://i.imgur.com/FNPg6LV.jpg)

필터는 하나만 존재하는 것이 아니며, 여러개 존재한다.

그 필터의 개수 만큼 activation map의 개수가 정해진다. 그리고 이 activation map은 output channel의 크기이다.

즉, 필터의 개수 = activation map의 개수 = output channel

![Image](https://i.imgur.com/S1uKrEC.jpg)

즉 위 그림으로 설명하자면, 3*32*32 input image에 6*3*5*5(3*5*5 filter 6개)를 가하면 그 output으로 6*28*28(6개의 1*28*28)의 output image가 발생한다.

bias도 고려하면 다음 그림처럼 된다.

![Image](https://i.imgur.com/5mfn7b3.jpg)

## activation map에 대한 두가지 해석

---

![Image](https://i.imgur.com/vpQGHn1.jpg)

1. 6의 activation map의 적층
    1. 해당 차원의 filter에 대해 얼마나 반응했는가 
2. 각 위치에 따른 6차원의 vector 막대기의 집합
    1. 6차원에서 한 차원마다의 의미 = 그 부근에서(Chunk에서) filter에 얼마나 반응했는가

## 다시 돌아가서 convolution

---

input 이미지도 하나가 아니라 batch 단위로 들어와서 연산을 할 수 있다.

![Image](https://i.imgur.com/IlAUeDQ.jpg)

이렇게.

이를 기호로 표현하면 다음과 같이 표현할 수 있다. 대응되는 관계를 잘 보면...

$C_{out}$은 filter의 개수가 결정한다.

$C_{in}$은 input image가 결정한다.

$N$은 image의 개수. 즉, batch size이다.

$H, W, H', W'$은 convolution을 거치면서 image의 크기가 변한다는 의미이다.

![Image](https://i.imgur.com/KdQdeTX.jpg)

즉 다음과 같이 Convolution 과정을 거치면 이미지가 다음과 같이 변할 것이다.

![Image](https://i.imgur.com/GKmRP2H.jpg)

참고로 Convolution도 dot product인지라 연산 자체는 linear하기에, Conv 두번 한 것도 결국엔 그저 다른 Conv하나일 뿐이다.

이를 개선하기 위해 Conv 다음에 ReLU를 넣는다.

![Image](https://i.imgur.com/6nWpgKi.jpg)

## 그래서 Conv filter가 학습하는 것이 무엇인가?

---

앞서 본 Linear Classifier와 Multi Layer Perceptron은 다음과 같은 input image를 학습했다.

![Image](https://i.imgur.com/jLEqI5z.jpg)

이 들의 문제는 앞서 말했듯 그냥 색깔만 나타나 있어서, 색만 비슷하면 같은 것이라고 인식하고, 형체도 불분명해 spatial structure를 잘 반영했다고 보기 어렵다는 것이 문제다.

...

하지만 First Layer Conv filter를 거친 image는 spatial structure를 잘 가지고 있는 template를 내보낸다.

이 template을 통해 보아하니, filter는 local에서의 oriented edge, wavelet, oppposing color(색의 반전)를 학습하고 있는 것 같다.

참고로 학습된 Conv filter는 고양이의 눈과 비슷하다고 한다. oriented edge를 잘 판단한다고...

![Image](https://i.imgur.com/fZBB7Bc.jpg)

# Convolution 연산 과정

---

![Image](https://i.imgur.com/VbFOwO5.jpg)

이렇게 하나씩 slide해가는 과정을 거친다.

그 결과 input은 7*7이었는데, output은 5*5이다.

...

보면 feature map이 계속 줄어드는데, 이는 큰 문제이다.

현재 general equation

input : $W$

filter: $K$

output : $W-K+1$

이를 해결하기 위해서 padding을 해서 해결한다.

보통은 zero를 이용해서 padding을 한다. 물론 평균값을 넣는 등 다양한 방법이 있다.

![Image](https://i.imgur.com/9BI5ywO.jpg)

현재 general equation

input : $W$

filter: $K$

padding : $P$

output : $W-K+1+2P$

same input-output condition : $P=(K-1)/2$

# Receptive Fields(Receptive Value of output tensor)

---

Receptive Fields는 크게 두가지 의미를 가진다.

1. Convolution 연산이 가해지는 input image내에서의 영역
    
    아래 그림의 경우 input image에 그려진 푸른 네모 영역인 것이다.
    

![Image](https://i.imgur.com/Z3k3ZJn.jpg)

이번에는 여러개의 Conv layer를 적층해본다.

그러면 Conv layer의 적층으로 인해,  최종 output value는 가장 오른쪽의 7*7 input image의 영향을 받게되는 것을 알 수 있다. 

![Image](https://i.imgur.com/z0Cu15F.jpg)

자세히 살펴보자.

![Image](https://i.imgur.com/8G96SR0.jpg)

output이 3*3 input을 받는다.

![Image](https://i.imgur.com/eWPj9iv.jpg)

output이 5*5 input을 받는다.

![Image](https://i.imgur.com/bJR7cGW.jpg)

output이 7*7 input을 받는다.

...

이에 따라 receptive field는 다음 두가지 의미를 지니게 된다.

1. receptive field in the previous layer(= of an activation) = kernel size convolution (kernel의 크기)
2. receptive field in the previous layer(= of an activation) = spatial size in the input image that affect to the value of the output (input image에서 output에 영향을 주게되는 지역)

이 그림을 볼 때, Conv layer를 쌓을 수록 receptive field의 크기가 linear하게 자라나는 것을 알 수 있다.

...

다만 이는 문제가 있는데, 큰 image를 보기 위해서는 많은 layer가 필요해진다.

이를 해결하기 위해서는 network 내부를 downsample해야한다. 이는 새로운 hyperparameter이다.

**이를 strid라고 한다.**

...

### 중간의 Question에 대한 답

---

zero padding은 information을 주려는 게 아니라 그저 image의 크기를 유지하려는 목적으로 사용된다.

다만 zero padding에 의미를 부여할 수 는 있기는 한데, 애초에 convolution은 image의 의미가 불변해야해서 그러면 안된다.

다만, zero padding을 넣고 convlayer를 거치면, filter가 어디에 input image가 있는지 학습하게 된다. 이런 의미에서는 추가 정보를 준다고 보는 게 맞을 지도

...

## Stride

---

큰 image를 보기 위해서 많은 layer를 쌓는 대신, stride를 늘린다.

한칸씩 뛰는 걸 두칸씩 뛴다는 의미이다.

downsample할 뿐만 아니라 receptive field를 더 빠르게 흝을 수 있다. → receptive field를 double한다.

![Image](https://i.imgur.com/d5kj9GA.jpg)

현재 general equation

input : $W$

filter: $K$

padding : $P$

stride : $S$

output : $(W-K+1+2P) / S + 1$

same input-output condition : $P=(K-1)/2$

만약 나눈 값이 나누어 떨어지지 않는다면 truncate한다.

## Example

---

![Image](https://i.imgur.com/lbmLNeq.jpg)

![Image](https://i.imgur.com/RMyZ2TC.jpg)

![Image](https://i.imgur.com/utaaPP7.jpg)

# $1 \times 1$ Convolution

---

하나하나 dot product하는 것과 같은 경우이다.

$1 \times 1$ Convolution + ReLU가 연속으로 있는 구조를 network in network structure라고 한다.

이는 Fully connected layer와 같은 기능을 하는데, 이미지의 각 위치마다(각 feature vector마다) independent하게 작동한다는 차이가 있다.

![Image](https://i.imgur.com/BMhnFAz.jpg)

### $1 \times 1$ Convolution와 Fully connected Layer의 차이

---

$1 \times 1$ Convolution

channel을 변경

spatial structure 유지 → channel만 변경하는 adapter로 사용.

Fully Connected Layer

flatten한 tensor를 product

destory spatial structure → category score를 구할 때 사용한다.

# Convolution Layer Summary

---

![Image](https://i.imgur.com/y1eO9h3.jpg)

보통은 다음 설정값을 이용하곤 한다.

![Image](https://i.imgur.com/t6BBkRk.jpg)

# Other Dimension Convolution0

---

그동안은 2D Convolution만 다뤘다.

![Image](https://i.imgur.com/7vUxEBF.jpg)

### 1D Convolution

---

text data, audio data 등에 사용

![Image](https://i.imgur.com/EWZy0IK.jpg)

### 3D Convolution

---

3D grid에 각각 feature vector(정확히는 tensor라고 하는 것이 맞을 듯)가 있는 것이다... 직사각형이 움직이면서 convolution 하는 것.

point cloud 등의 3D data에 사용한다.

![Image](https://i.imgur.com/smKXEtv.jpg)

# Pooling Layer

---

이미지를 downsample하는 데 사용된다. learnable parameter는 없다.

컨벌루션에서도 stride 1 초과로 주면 downsample 가능하지만, 이를 대신해서 pooling이 가능하다.

![Image](https://i.imgur.com/TMQ68Ob.jpg)

이미지의 각 region마다 pooling을 거친다. 이는 모든 channel에 가해진다.

## MaxPooling

---

![Image](https://i.imgur.com/VQxk6j5.jpg)

기본적으로는 위와 같이 kernel size = stride로 두어서 kernel이 overlap하지 않게한다.

그리고 kernel이 겹친 공간에서 max 함수를 사용해서 가장 큰 수만을 남긴다.

1. learnable parameter가 없고, 2. spatial invariance가 어느 정도 보장되어서 자주 사용된다.

Maxpooling이 nonlinear한데, 왜 ReLU가 도입되는가?

금발근육남께서는 반드시 필요하다고 생각하지는 않는다. 실제로 사용하지 않는 논문도 있다고.

다만 보통은 Convolution 뒤에 ReLU를 넣는다고 한다.

## Pooling Summary

---

![Image](https://i.imgur.com/37PXPiq.jpg)

![Image](https://i.imgur.com/nq6gYVu.jpg)

# Convolution Network의 형태

---

Fully-Connected Layer, Activation Function, Convolution Layer, Pooling Layer를 모두 합쳐서 만든다.

![Image](https://i.imgur.com/bobIfOs.jpg)

## Example : LeNet-5

---

![Image](https://i.imgur.com/zVdWLPI.jpg)

layer를 지나갈 수록, spatial size는 줄어들고, channel은 증가한다. → volume이 유지된다!

...

하지만 deep 해질 수록 net을 train하기 더 어려워진다.. 이를 해결하기 위해 Normalisation을 도입한다.

# Batch Normalisation

---

input을 받아서 그 output을 평균 0, 분산 1의 분포로 만드는 Layer이다.

이는 internal covariant shift 때문이다.

여러 개의 Layer를 거치면서, 다음 layer에 관측했을 때, 직전 layer의 output의 데이터 분포가 input과 다르게 shift했다는 것

![Image](https://i.imgur.com/tpORoCg.jpg)

그래서 모든 output을 분포를 통일 시켜버리겠다는 뜻

![Image](https://i.imgur.com/5EHJWHp.jpg)

다음 공식을 따른다.

여기서 $x^{(k)}$는 sample이다.

![Image](https://i.imgur.com/VThd2hx.jpg)

![Image](https://i.imgur.com/MijAGjF.jpg)

$\epsilon$은 0으로 나누는 걸 방지하기 위해 도입되었다.

하지만 모든 layer의 output을 평균 0, 분산 1로 만들기 어려울 수 있다.

그래서 batch normalisation 이후 다음 연산을 진행한다.

![Image](https://i.imgur.com/eib8Dw7.jpg)

$\gamma, \beta$의 차원은 $D$이다.

layer가 vector의 각 element마다 어떤 mean과 variance가 적합한지 찾게 한다.

$\gamma = \sigma, \beta = \mu$일때, batch normalisation은 identity function이 된다.

...

또다른 문제도 있다. $\sigma, \mu$는 minibatch에 dependent한 속성이다. 지금까지 이런 속성은 없었다.

$\sigma, \mu$는 minibatch마다 다른데, 이를 test하는 동안은 측정할 수가 없을 뿐더러, batch의 분포마다 prediction의 결과가 달라지는 문제가 발생할 수 있다.

따라서 다음과 같이 계산한다.

training 중 : 원래대로 구한다.

![Image](https://i.imgur.com/MAlariW.jpg)

test 중 : training 당시 구했던 것을 그대로 사용한다.

![Image](https://i.imgur.com/3cyvnAA.jpg)

따라서, batch norm을 test하는 동안, 이는 linear opeartor가 된다.  → test time에는 free한 layer가 된다.

Batch Normalisation Layer는 Fully Connected Layer나 Convolution Layer와 합쳐질 수도 있다.

## Fully Connected Net, ConvNet에서 Batch Normalization

---

![Image](https://i.imgur.com/II40SH2.jpg)

## 사용법

---

보통 Fully Connected Layer 혹은 Convolution Layer 뒤, 그리고 nonlinearity 앞에 온다.

![Image](https://i.imgur.com/Y123Act.jpg)

train이 빨라지는 등 여러 장점이 있다.

![Image](https://i.imgur.com/5Pj9ASP.jpg)

하지만 다음과 같은 문제가 있다.

train/test 동안 batch norm을 키고 꺼야하는 문제도 있고,

dataset이 imbalance한 경우에는 batch norm이 잘 안맞는 경향이 있다.

![Image](https://i.imgur.com/z1vmuNX.jpg)
## Layer Normalization, Instance Normalization

---

이를 해결하기 위해 다음과 같은 방법을 도입한다.

batch dimension 뿐만이 아니라 feature dimension에 대해서도 평균과 분산을 구한다.

즉, batch에 따라서 영향을 안받게 되어서, train과 test를 가리지 않고 사용할 수 있게 된다.

![Image](https://i.imgur.com/5hFoDje.jpg)

이미지에서는 이걸 더 많이 쓴다.

spatial dimension에 대해서 평균과 분산을 구하는 것이다.

![Image](https://i.imgur.com/SkgICXr.jpg)

batch norm : batch, spatial dimension에서 평균 분산

layer norm : spatial, channel dimension에서 평균 분산

instance norm : channel dimension에서 평균 분산 (그림으로는 spatial dimension에서 평균 분산아닌가)

![Image](https://i.imgur.com/3ICZPRZ.jpg)

Group Normalization은 channel 몇개를 group으로 만들어서 그리고 그 channel 사이에서만 평균 분산을 구한다.

![Image](https://i.imgur.com/Y2D8zjG.jpg)

# My Own Question

---

## oriented edge

---

## wavelet

---

## opposing color

---

### downsample

---