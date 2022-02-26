# Lecture 18 : Video

---

dimension을 시간으로 확장하는 것이다

![](https://i.imgur.com/ARt8tWI.png)

![](https://i.imgur.com/3bBIzZM.png)

![](https://i.imgur.com/HEMLTkq.png)

이미지가 너무 커서 FPS를 줄이고 해상도도 줄였다.

![](https://i.imgur.com/R0y7cqB.png)

training은 부분적인 영상으로 하고, test는 여러 부분에 대해서 input을 준 다음, 그 output의 평균을 내서 구한다.

![](https://i.imgur.com/P4lIVSR.png)

# Video Classification

---

# Single Frame CNN

---

가장 간단하고 확실한 방법이다. 하지만 temporal structure가 무시된다.

그저 하나의 frame마다 train하고 test할 때는 이를 평균내는 것이다.

![](https://i.imgur.com/VmXcCY3.png)

# Late Fusion

---

아래 두가지는 Late fusion이다. 각 frame을 CNN에 넣고, flatten/average pool해서 Class에 맞게 combine한다.

![](https://i.imgur.com/bkQNH40.png)

![](https://i.imgur.com/j9uaB8z.png)

# Early Fusion

---

다만, 이러면 근처 픽셀의 data 변화를 파악하기 어렵다. 여러 픽셀을 하나의 vector에 combine하기 때문이다.

이를 해결하기 위해 Fusion을 초장부터 하는 경우도 있다.

하지만 여전히 문제인 것이, Temporal dimension을 너무 일찍 붕괴시켜서 frame 간의 구분이 어려워진다.

![](https://i.imgur.com/zgCj0Dj.png)

# 3D Fusion

---

그래서 3D CNN과 pooling을 한다.

fusion이 느리고, temporal dimension이 살아있다.

![](https://i.imgur.com/Gt04CPJ.png)

# 종합하면..

---

앞서 보인 문제점과는 별개로,

위 방법론들은 temporal extent에 대해서 receptive field가 커지게 된다.

spatial receptive field도 마찬가지이다.

특히 global avg할 때 그 효과가 두드러진다.

![](https://i.imgur.com/N8FZ7KS.png)

![](https://i.imgur.com/ff6kxoa.png)

이때  2D와 3D의 Convolution이 무슨 차이가 있을까?

![](https://i.imgur.com/SSbULl1.png)

# 2D Conv & 3D Conv

---

2D convolution은 그 filter의 시간 차원의 길이가 input의 시간차원 길이로 고정되어있다.

그래서 output이 3차원으로 나온다.

![](https://i.imgur.com/XuiMzuR.png)

이로인해서, 각 시간마다 서로 다른 filter를 생산해줘야하는 문제가 생긴다.

![](https://i.imgur.com/xUactSb.png)

반면, 3D Conv의 경우는 시간축에 대해서도 Convolution slide가 가능하다.

![](https://i.imgur.com/U3Q1o3k.png)

그래서 하나의 filter만 train해도 되고, output도 시간 차원이 살아있다.

![](https://i.imgur.com/6qGBOuq.png)

첫 layer filter는 다음과 같이 움직이는 edge 영상을 띄게 된다.

![](https://i.imgur.com/mvgCBja.png)

성능은 더 나은 것을 알 수 있다.

![](https://i.imgur.com/geocGtE.png)

이제 이 3D CNN으로 VGG와 같은 형식화된 모델을 만든다. 대표적으로 C3D

![](https://i.imgur.com/C7IQlGc.png)

성능도 좋다.

![](https://i.imgur.com/vXJYruI.png)

# Motion

---

3D convolution은 space와 time을 동일한 느낌의 차원으로 취급한다. interchangable하다는 것이 그 증거.

이제는 motion만 따로 추출함으로써, space와 time을 개별적인 차원으로 보려고한다.

![](https://i.imgur.com/h34wWP8.png)

서로 1 프레임 차이나는 두 개의 프레임을 가져와서 flow field를 만든다.

여기서 horizontal 방향과 vertical 방향 두가지를 만든다.

![](https://i.imgur.com/3shOm2X.png)

visual appearnce는 각각의 frame을 따로 떼어내서 spatial stream convnet으로 보내고,

flow field는 모두 early fusion한 후 temporal stram convnet으로 보낸다.

![](https://i.imgur.com/nSYtfNb.png)

성능이 상승한 것을 볼 수 있다.

![](https://i.imgur.com/V2ryoGa.png)

# Modeling long-term temporal structure

---

방금전까지 살펴본 3D CNN과 같은 방법은 길어봐야 5초 정도의 clip으로 학습하는 것이다.

이 input 길이를 늘리기 위해서 RNN을 도입한다

![](https://i.imgur.com/uhyzi0m.png)

many to one 구조에서는 single classification을 한다고 볼 수 있다.

![](https://i.imgur.com/wlEl59n.png)

many to many 구조는 video에서 프레임마다 caption을 가져오는 일에 쓸 수 있다.

![](https://i.imgur.com/5m7Tb9o.png)

이 모델에서 CNN은 pretrained 모델을 사용하면 RNN만 backpropagate하면 된다!

![](https://i.imgur.com/HizyR6D.png)

이와 같은 구조는 CNN은 한 부근의 시점에 대해서 학습을 하고, RNN은 전체적인 시점에 대해서 학습을 한다고 볼 수 있다.

![](https://i.imgur.com/66xV2Oq.png)

이 두 접근법을 Multi Layer RNN을 이용해서 융합한다

![](https://i.imgur.com/oVFxwUw.png)

각 block은 2D feature map이다.

각 block은 same layer, previous timestep과 previous layer, same timestep 두 가지 input에 영향을 받는다.

이 각각의 feature들은 Convolution 연산을 이용해서 fusion한다.

각 Layer마다 서로 다른 weight을 사용한다.

![](https://i.imgur.com/eM0FRcv.png)

어떻게 하는지 면밀히 보자.

2D Conv를 하면 다음과 같다.

![](https://i.imgur.com/Ay8S53T.png)

이렇게 Recurrent Convolution Network에서 feature fusion으로 convolution을 한다고 하면, 공식을 세워야한다.

![](https://i.imgur.com/Of8H2IX.png)

이렇게 Vanilla RNN 공식을 쓰면, Weight 대신에 Convolution을 하고 그 결과를 더한 후 tanh를 거치면 된다.

![](https://i.imgur.com/AKzpyC2.png)

LSTM, GRU 등 다양한 RNN 모델에 넣을 수 있다.

![](https://i.imgur.com/cVwrSFn.png)

하지만 long sequence에 대해서 느리다는 문제가 있다. 다음꺼 계산하려면 전의 것이 계산되어있어야해서, 병렬연산이 안된다는 것

![](https://i.imgur.com/fgxMZXH.png)

# Spatio-Temporal Self - Attention

---

앞서 본 방법들. 병렬연산이 쉽다는 점까지 해서 self attention이 좋다.

![](https://i.imgur.com/InVt4Qu.png)

예전에 보았던 self attention에 관한 내용. 여기서 affinity matrix가 E, A를 의미한다.

![](https://i.imgur.com/Qj26nUT.png)

Nonlocal Block에서 초기화 할때 last Convolution filter를 0으로 초기화한 후, pretrained 된 model에 넣어서 fine tuning할 수 있다.

그러면 residual connection을 통해서 마지막 convolution filter가 잘 학습된다.

![](https://i.imgur.com/mDPOdQ0.png)

이렇게 Nonlocal Block은 global한 fusion을 3D CNN은 slow local fusion을 담당한다.

![](https://i.imgur.com/3CLfPIN.png)

# I3D : Inflating 2D Networks to 3D

---

그럼 최고의 3D CNN 모델은 무엇일까? 

대표적인 2D CNN 모델을 3D로 확장한 모델이 보통 성능이 좋다고한다.

![](https://i.imgur.com/oPaLJ6b.png)

이 때, 2D conv filter를 K_t개 만큼 복사(time dimension만큼)해서 3D conv로 확장한다. 그리고 그 수치를 K_t만큼 나눈다.

이런 방식은 같은 이미지를 틀어놓는 영상에 2D convolution을 하는 것과 같은 효과를 낸다고 한다.

![](https://i.imgur.com/bd8PwY4.png)

성능은 더 좋아졌다.

![](https://i.imgur.com/quPsfm9.png)

# Visualizing Video Models

---

이제 저번에 했던 것처럼, image와 flow field를 추출하는 과정을 얻어보자.

![](https://i.imgur.com/fq4zQxn.png)

![](https://i.imgur.com/XjeXhQo.png)

![](https://i.imgur.com/wlyd4Ph.png)

# SlowFast Network : Time & Space 구분

---

SlowFast Net을 이용하면 low FPS, many channel과 high FPS, low channel을 융합할 수 있다.

중요하지 않으므로 자세한 건 생략

![](https://i.imgur.com/QJKIomW.png)

![](https://i.imgur.com/3qoi6cU.png)

# Spatio-Temporal Detection

---

frame마다 달라지는 action을 구분하고 싶다.

![](https://i.imgur.com/nTke2cF.png)

Faster R-CNN과 유사한 구조를 통해서 temporal proposal을 먼저하고, 해당 proposal을 대상으로 classifying을 한다.

![](https://i.imgur.com/urpCXEn.png)
