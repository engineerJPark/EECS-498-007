# Lecture 20 : Generative Models 2

- 앞서 이런 걸 했다.

![](https://i.imgur.com/pxf7Qhp.png)

![](https://i.imgur.com/lfCD4mo.png)

![](https://i.imgur.com/4yyLXC9.png)

![](https://i.imgur.com/Yav2b2R.png)

![](https://i.imgur.com/XqyShMG.png)

![](https://i.imgur.com/FuZK0lq.png)

![](https://i.imgur.com/bZGTbZX.png)

![](https://i.imgur.com/YS3Tl3t.png)

- VAE의 다른 모델과의 차이점은 Likelihood of data와 함께 latent vector z를 구한다는 것이다.

- 그리고, Encoder Decoder 모두 lower bound를 구하는데 사용했다.
  
  - 이 두 넷을 학습해서 lower bound를 최대로 만든다.

- Encoder는 구체적인 이미지 x에서 확률분포를 output하는데, 이는 다른 모델에서는 볼 수 없던 것이다.
  
  - 이럴려면 트릭을 써야하는데, diagonal gaussian을 쓰고, mean과 Diagonal covariance를 output해야한다.

# Fully Connected Layer Variational Autoencoders

- 여기서부터는 실제로 VAE를 구현한 예시이다.
  
  - MNIST 예제를 푸는 모델을 만든 것

- 아래와 같이 input과 output 사이에 Fully Connected Layer를 둘 수 있다.

- 여기서 출력 dimension 20은 hyperparameter로, Diagonal Gaussian의 dimension이다.

- Covariance는 원래 Diagonal Gaussian의 dimension의 제곱인 dimension을 가져야하지만, trick에 의해서 Diagonal Covariance를 쓰게 됨으로써 똑같은 크기의 dimension을 쓰게 되었다.

![](https://i.imgur.com/XFNNVHX.png)

## Training

- Encoder
  
  - output probability distribution over latent variable z
  
  - 이 output을 variational lower bound의 KL divergence를 계산하는 데 사용한다.
    
    - 여기서 $q_\phi(z|x)$가 바로 Encoder가 output으로 놓는 **Diagonal Gaussian**이다. 
    
    - $p(z|x)$는 **simple distribution(unit gaussian)**. 이것은 학습 대상이 아니다. training 시작 시점에서는 z가 고정이기 때문.
  
  - 위의 두 입력이 Gaussian이면 아래의 수식과 같이 'Closed Form'으로 KL Divergence를 계산할 수 있다.
  
  - 다른 dataset이더라도 동일한 prior distribution을 사용해도 괜찮다.
    
    - prior는 latent variable z에 대하여 구성되어있다. 이는 dataset에서 관측 불가하다. 따라서,  model은 latent variable을 그 어떤 것과도 연관성 없이 학습하게 된다.
    
    - 즉, 특정한 prior를 고른다는 것은 model에게 어떤 latent variable을 학습할지 전달하는 것과 같은 효과를 띄게 된다.
    
    - 즉, Gaussian(diagonal)을 전달하면 어떤 data에도 independent하게 latent variable을 학습하라고 전달하는 것이 된다. 그리고 0 mean 1 variance를 띄게 된다.
    
    - 따라서 다른 dataset이더라도 동일한 prior distribution을 사용해도 괜찮다.

- Diagonal Gaussian 대신에 (dimension of z) classifier를 학습시켜도 되는가
  
  - 하지만 Encoder network 내부에서 연산을 공유해야한다. 
  
  - 2 level modeling이라고 하는데,
    
    - 하나는 many layers를 계산하는 Neural Network
    
    - 하나는 probabilistic formulation이다.
  
  - 그래서 만약 latent variable을 uncorrelated하게 학습하려고 한다고 하면, mean과 standard deviation of z은 각 layer에서 많은 parameter와 weight을 공유해서 그 결과가 나오게 된다.

![](https://i.imgur.com/tKO2yxn.png)

- 예측한 distribution으로부터 sampling을 해서 concrete한 z를 구한다.

- 그리고 이제는 distribution over the images x를 구한다.

![](https://i.imgur.com/nzEfStC.png)

- 그 후 first term을 구할 수 있다.
  
  - 이제 기대값을 구하는데, 그 기대값을 구하는 대상 분포가 방금 구한 z의 diagonal Gaussian이다.
  
  - 기대값 내부에서는 data reconstruction term이 들어간다.
    
    - predicted $\hat x$와 input $x$가 비슷해야한다는 의미이다.
  
  - 이렇게 해서 Likelihood를 최대로 만드는 mean과 Diagonal covariance를 구한다.
    
    - 여기서 두 번째 KL Divergence가 구하기 쉬워진다. decoder 결과를 통해서 predicted data의 likelihood를 maximize한다고 하자.
    
    - 그러고 나면 이제 second term은 구하기 너무 쉽다. x|z의 distribution만 주어지면 쉬워진다.

![](C:\Users\jshac\AppData\Roaming\marktext\images\2022-03-01-20-23-35-image.png)

- 이렇게 loss에 있는 두 개의 term의 값을 Encoder와 Decoder를 통해서 구하고, jointly train할 수 있다.

- 참고로 이 두 항은 서로 배척 상태에 있다.
  
  - 첫 항은 data reconstruction term
    
    - data input -> latent -> data prediction이 원활해지게 하는 term
    - latent code가 많은 정보를 함축할 수 있도록 유도한다.
  
  - 둘째 항은 KL Divergence between prior & samples from Encoder(Diagonal Gaussian)
    
    - predicted distribution over latent 는 간단해야한다는 의미(Gaussian)
    
    - Encoder가 predict할 수 있는 type에 제한을 걸어놓는 것이다. prior와 비슷한 형태만 되도록!

![](https://i.imgur.com/sRwAiyf.png)

# Variational Autoencoders : Generating Data

- 학습한 Autoencoder를 이용해서 신기한 것을 해볼 것이다.
  
  - Decoder만 사용할 것이다.
  
  - prior distribution $p(z)$를 추출한다. 그리고 decoder에 먹인다.
  
  - distribution over data $x$를 구한다.
  
  - 이걸 또 sample해서 $\hat x$를 구한다.

- 이제는 진정한? result를 볼 수 있다.

![](https://i.imgur.com/B4GBNEs.png)

- 이것이 바로 예시. 진짜 generating data 중이다.

![](https://i.imgur.com/j8q0ebo.png)

- latent가 prior distribution과 비슷해지도록 제한을 걸어놨었다.
  
  - prior가 diagonal gaussian : each latent variable은 independent하다.
  
  - 아래 그림처럼 서로 다른 latent로 decoder에 넣으면 다른 이미지가 나온다.

- **단순히 generating을 학습하는 것이 아니라, 이 latent code를 학습함을 통해서 generate하는 것이다.**

- **latent z를 조작하면 generating image를 조작할 수 있다는 것**

- 이런 건 autogressive model은 할 수 없는 것

![](https://i.imgur.com/gfYW4pO.png)

## Editing Images

- Image를 넣고 수정하는 것이다.
  
  - train한다.
  
  - test time에서, 수정할 대상인 x를 입력받고, latent code z를 출력.
  
  - latent code를 수정한다.
  
  - 이걸로 $\hat x$를 출력한다.

- latent가 higher order structure in the data를 담는다.

- generator가 그것을 찾는다. lower bound를 maximizing해서

![](https://i.imgur.com/geSKXzr.png)

- 이걸 활용한 예시

- 물론 어떤 latent code가 어떤 특성의 변화에 영향을 주는 지는 알 수 없다. 이건 model만 알 수 있다.

- 알아보려면 실제로 이렇게 plot을 해봐야 알 수 있다.

![](https://i.imgur.com/IEF6vjO.png)

![](https://i.imgur.com/cZMXKFZ.png)

- 하지만 아직 likelihood를 직접 조작하지 못하고, sampling 결과가 blurry하다는 문제가 남아있다.

- 

![](https://i.imgur.com/ziUAZ0i.png)

- 지금까지 한 것

- 이 둘을 합친다.

![](https://i.imgur.com/lNF2j1Z.png)

# Vector-Quantized Variational Autoencoder(VQ-VAE2)

- 깊게는 다루지 않는다.

- Conditional Generative Model이다.

- 운용 순서
  
  - variational autoencoder type method를 학습한다. latent grid(not vector)를 학습하게 된다.
  
  - Pixel CNN : raw image가 아닌, latent grid에 적용해서 sampling

![](https://i.imgur.com/o3Xvs0N.png)

- 효과가 좋다고 한다.

![](https://i.imgur.com/9Uzp4Cu.png)

![](https://i.imgur.com/u8FoJdk.png)

# Generative Adversarial Network

- 앞서 본 model의 likelihood를 보자.

- 그리고 GAN(Generative Adversarial Network)의 likelihood를 보자.
  
  - 이 방법론은 density function of images를 modeling하는 것을 과감히 포기했다.
  
  - 그리고 오로지 sampling에만 집중한다.

![](https://i.imgur.com/bS88mHl.png)

- 방법론
  
  - **likelihood는 관심없고, 오로지 sample에만 집중**
  
  - 우선 변수 목록
    
    - x_i : training data, p_data로 부터 추출됨
    
    - p_data : training data의 true probability distribution
      
      - density function of nature. evaluate하거나 write down할 수 없다.
      
      - 그냥 image가 이 distribution으로부터 sampled 되었다고 믿는 수밖에 없다.
  
  - latent variable z를 이용한다.
    
    - fixed prior p(z)를 이용해서 구한다. (uniform distribution, diagonal gaussian)
    
    - z를 p(z)로부터 sample하고, Generator Network G(z) = x에 넣는다.
    
    - x는 그 결과 나온 sample of data
    
    - Generator Network는 probability distribution을 만든다. 이를 p_G라고 한다.
    
    - **목적은 Generator Network를 학습 시켜서 p_G와 p_data가 같게 하는 것이다.**
  
  - $p(z)$ -> $z$ -> $p_g$ -> sampling image ... 의 순서를 거치게 된다.
  
  - Discriminator net으로 pg pdata가 같게 한다.
    
    - 이 넷은 약간 Classification 역할을 한다.
    
    - 입력된 이미지가 진짜인지 가짜인지 판단한다.
    
    - 즉, sampled image와 real image 둘로 인해 학습된다
    
    - 이 과정은 supervised learning이다.
  
  - 그리고 이 두 네트워크를 jointly 학습한다.
  
  - Discriminator는 입력된 이미지가 fake인 부분을 학습하고, Generator는 생성하는 이미지가 진짜라고 인식하게 학습한다
  
  - 두 네트워크 모두 잘 학습되면, pdata와 pg는 근접하고, Generator에서 만드는 이미지가 진짜와 매우 흡사하게 된다

![](https://i.imgur.com/FHNRB0b.png)

- 수식의 의미를 파악해보자. 근본적으로 아래 식은 Cross Entropy이다.

- D는 전체 식을 최대로 하려고하고, G는 전체 식을 최소로 하려고 한다.

- 첫번째 항
  
  - p_data에서 추출한 x(real image, training set)에 대하여 기대값
  
  - Discriminator는 이 term을 최대로 하고 싶다.
  
  - D(x)는 0부터 1 사이의 수이다. 그리고 log는 단조함수이다.
    
    - 최대가 되려면 D(x) = 1이어야하고, 이는 real data임을 나타낸다.
  
  - 첫 번째 항의 의미는 Discriminator가 진짜 데이터를 진짜라고 판별하려고 한다는 의미이다

- 두번째 항은 반대이다. 가짜를 가짜로 판별하려한다는 의미
  
  - p(z)에서 추출한 z(latent variable)에 대한 기대값
  
  - Discriminator는 이 term을 최대로 하고 싶다. 그래서 D(x) = 0이 되려고 한다. 이는 x가 fake data임을 나타낸다.
  
  - 즉, fake data가 fake로 classified 되도록 유도하는 항이다.

- 하지만 두번째항을 Generator 입장에서 본다면 두번째 항을 minimizing하려고 한다.
  
  - 즉, D(x) = 1이 되도록 바란다.
  
  - 즉, Generator는 generated image가 real로 분류되도록 조정된다.

![](https://i.imgur.com/VWH5cGp.png)

- 하나는 Gradient Ascent 하나는 Descent를 한다
  
  - 원하는 Loss의 형태가 서로 Max min이라서...

- V(G, D)로 간단하게 표현한다.

- 하지만 문제가 있다.
  
  - 전체적인 Loss가 떨어진다고 모든 것이 해결된 것이 아니다.
  
  - Generator와 Discriminator 각각의 Loss가 모두 떨어져야 의미가 있다.
  
  - 그리고 단조감소 함수형태로 떨어지지 않고, 되게 복잡한 형태로 하락한다고 한다.

![](https://i.imgur.com/xlwZj5Z.png)

- 다른 문제도 있다.
  
  - 두번째 항을 아래와 같이 plot한다.
  
  - 처음 학습이 시작하는 부분에서 D(G(z)) = 0인데, 그래프를 보면 이 구간에서는 gradient가 0인 것을 알 수 있다.

- 그래서 G를 학습할 때, log(1-D(G(z)))를 최소화하는 것이 아니라 -log(D(G(z)))를 최대화 하도록 학습한다. 그러면 초기 gradient가 높게 나오게 된다.

![](https://i.imgur.com/GUO73ll.png)

![](https://i.imgur.com/MfC1rIA.png)

## Checking Optimality

- 연구자들의 말에 따르면, 이 minmax eq가 $p_G = p_{data}$일 때 global minimum을 얻는다고 한다.

![](https://i.imgur.com/YG6KM8A.png)

- 이를 검증해보자.

- $p(z)$를 $p_G$로 바꾸고 sampling 결과도 $z$에서 $x$가 나오도록 한다. $p_G = p_{data}$이니까 타당하다.

![](https://i.imgur.com/E843AaL.png)

- 이를적분하면 다음과 같은 형태로 나온다.

![](https://i.imgur.com/DwNqSrN.png)

- max 를 내부로 넣는다.

![](https://i.imgur.com/q3z4fvC.png)

- max를 계산한다.

- 각각의 x의 값에 따른 discriminator의 optimal value를 알아야 최대값을 구할 수 있다.

![](https://i.imgur.com/gQZdqZd.png)

- 다음과 같이 계산해서 형태를 구해보았다.

- 그 결과 아래 필기의 optimal discriminator(Loss 내부의 integral을 최대로하는)로 구할 수 있다.

- Optimal Discriminator는 Generator에 의존한다.

- 하지만 이 Optimal Discriminator의 값을 바로 구할 수 없다. $p_{data}, \space p_G$를 모르니깐

![](https://i.imgur.com/sRkCMfE.png)

- max를 지우고 $D(x)$에 $D^*(x)$를 넣는다.

- 그리고 $D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_{G}(x)}$를 대입한다.

![](https://i.imgur.com/OneMsQ8.png)

- 앞선 결과를 정리해서 다시 기대값 형태로 만든다.

![](https://i.imgur.com/dqe3zmi.png)

- 그리고 다음과 같이 정리의 시간.

![](https://i.imgur.com/7L9ljSk.png)

![](https://i.imgur.com/lpU54g1.png)

- KL Divergence는 두 확률 분포간의 거리를 의미한다.

![](https://i.imgur.com/NKSIHUp.png)

- Jensen-Shannon Divergence도 두 확률분포간의 거리를 의미한다.

![](https://i.imgur.com/Lh3GTb0.png)

- JSD는 항상 양수이고, 두 확률분포가 같을 때 0이된다.
  
  - 따라서 unique minimizer of this expression은 $p_G = p_{data}$일 때 발생한다.

![](https://i.imgur.com/WCmhdK4.png)

- GAN의 문제점
  
  - Structure가 고정이라서, weight만 변경이 가능하다. 따라서 Neural Network가 expressible function의 영역밖일 수도 있다. 즉, 진짜 Optimal한 G, D인지 판단할 수 없다
  
  - Optimal Solution의 수렴성에 대해서는 알 수가 없다

![](https://i.imgur.com/YabCI9b.png)

- 아래는 Nearest Neighbor를 한 행이 모아놓은 것. latent code 기반으로 생성된다는 것을 알 수 있다.

![](https://i.imgur.com/YGnRm7V.png)

## DC-GAN

- GAN을 발전시켜서 나온 모델

- Generator와 Discriminator 모두 Convolution을 이용

![](https://i.imgur.com/pS2gJSf.png)

![](https://i.imgur.com/9LeSbX2.png)

## Interpolation

- Z끼리 interpolation해서 새로운 이미지를 만드는 것이 가능하다.

- 맨 아래 행이 interpolation 결과 나온 것

- 단순히 alpha transparency blend가 아니다. 각 spatial structure를 서로 다른 spatial structure로 warp하는 것이다.

![](https://i.imgur.com/EKw7avj.png)

## Vector Math

- latent vector를 대상으로 vector Math를 해서 이미지를 새로 구할 수 있다.

- sample을 여러개 가져와서 아래와 같이 카테고리로 나누고, 이를 더하고 빼고 할 수 있다.

![](https://i.imgur.com/Cr4Nftr.png)

![](https://i.imgur.com/KI9zPsy.png)

# GAN Improvements

- 그리고 GAN 연구는 인기가 폭발! 그동안 발전한 것을 알아보자.

## Improved Loss Function

![](https://i.imgur.com/3yDMHXF.png)

## Higher Resolution

- 지금 기준으로도 이 정도면 큰 해상도 아닌가?

![](https://i.imgur.com/T1SbZso.png)

![](https://i.imgur.com/XRMVXq5.png)

- 참고로 이 Style GAN의 이미지 생성 및 변화 과정을 보면 deformation이 자연스러운데, 이는 training data를 기억하는 것이 아니라, generated image의 주요 structure를 기억하는 것임을 의미한다.
  
  - 이는 각 생성 이미지 간의 interpolation을 통해서 볼 수 있는 애니메이션이다.

# Conditional GAN

- 앞서 본 GAN은 모두 unconditional하다.
  
  - dataset을 기준으로 학습시키고, 새로운 이미지 sample도 dataset에서 해냈다

- Conditional GAN을 통해서 생성하는 이미지의 type에 대해서 더 제어할 수 있다.
  
  - image x는 label y로 conditioned
  
  - random noise c를 label y와 함께 input한다.

![](https://i.imgur.com/M3XGveX.png)

## Conditional Batch Normalization

- 최근에는 label y를 GAN에 넣는 방법을 **Conditional Batch Normalization**을 채택한다.

- 기존과는 다르게, 각각의 label class에 대해서 $\gamma, \beta$를 따로 학습하도록 한다

![](https://i.imgur.com/rP69nDY.png)

- 위의 Conditional Batch Normalization을 이용하면 Conditional GANs를 train할 수 있다. 
  
  - 이제는 random noise c를 넣지 않고, model에 어떤 category를 생성할 지를 전달한다. (Conditional Batch Normalization을 이용한다면 이렇게 하는 것인가??)
  
  - 다음과 같은 발전을 볼 수 있다. 전부 다 같은 모델에서 나온 것이다.

- spectral normalization은 다루지 않는다.

![](https://i.imgur.com/Nv8tHJ6.png)

- Self Attention을 사용하면 더 좋은 결과를 낼 수 있다.

![](https://i.imgur.com/vI7NXP1.png) 

- 2019년 기준 SOTA

![](https://i.imgur.com/FSW4xzX.png)

- GAN으로 video를 생성할 수도 있다.

![](https://i.imgur.com/5eV40aO.png) 

- 이번에는 Condition에 label이 아니라 아예 Text를 넣는다.

![](https://i.imgur.com/VsJq60E.png)

- 이번에는 condition에 이미지를 넣는다. 그러면 output x가 realistic한 high resolution image가 나온다.

![](https://i.imgur.com/hNftI98.png)

- GAN을 이용해서 image editing

![](https://i.imgur.com/zwEKd2l.png)

- input y = horse, output x = zebra

![](https://i.imgur.com/smWvLoH.png)

![](https://i.imgur.com/Q868hg8.png)

- convert label maps to image

- Style할 이미지와, Label Map을 한 이미지를 넣으면 다음과 같은 결과가 나온다.

![](https://i.imgur.com/IJfipzv.png)

- 어떤 데이터도 생성 가능하다.

- 경로 예측도 가능하다.

- condition y = 사람들이 지나간 경로, output x = 사람들이 다음에 지나갈 경로

![](https://i.imgur.com/Zy5gL6F.png)

# Summary

![](https://i.imgur.com/2QNXETo.png)

![](https://i.imgur.com/JDmzQsi.png)

![](https://i.imgur.com/TyplK1a.png)
