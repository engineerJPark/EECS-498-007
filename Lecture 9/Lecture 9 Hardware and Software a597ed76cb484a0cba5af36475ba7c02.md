# Lecture 9 : Hardware and Software

# CPU vs GPU

---

연산양이 많으면 GPU를 쓰는 것이 최고다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled.png)

# GPU 구조

---

메모리와 프로세서가 같이 있다. 하나의 작은 컴퓨터이다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%201.png)

72개의 스트리밍 멀티 프로세서로 구성되어있다. 이를 SMs라고도 한다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%202.png)

하나의 SM을 들여다보면, floating32의 core가 64개 박혀있는 것을 볼 수 있다.

그래서 실수 연산을 하면 다음과 같은 FLOP인 것을 알 수 있다.

참고로 GPU는 한 사이클에 곱하기 더하기를 한 번씩 할 수 있어서 2 FLOP/cycle이다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%203.png)

Tensor core는 NVIDIA가 matrix multiply 특화용 연산장치를 만든 것. 최대 4*4연산이 가능하다.

FLOP은 4*4*(2*4-1) + 16 = 128

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%204.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%205.png)

Matrix를 4*4로 나눠서 계산한다. 이것보다 더 차원이 큰 곱은 다른 tensor core에서 padding을 해서 사용한다.

즉 무조건 4*4로 행렬곱이 발생하므로, 모든 계산은 2의 승수로 계산하는 것이 좋다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%206.png)

# Software

---

기본적으로 DL framework는 다음 세가지를 지원한다.

1. 빠른 prototyping
2. Auto computing gradient
3. Run in GPU/TPU

# PyTorch

---

큰 구성 요소는 다음 세가지이다.

1. Tensor
2. Autogrtad
3. Module

## Tensor

---

아래는 사용 예시이다. GPU 쓸거면 맨 윗줄 그냥 ‘cuda’로 바꾸면 된다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%207.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%208.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%209.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2010.png)

## Autograd

---

autograd를 사용하는 경우, requires_grad=True를 tensor 선언시 해준다. 그래야 그 텐서가 computational graph에 등록된다.

 requires_grad=True인 텐서로 연산한 결과는 항상  requires_grad=True이다. 자동으로 computational graph에 등록된다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2011.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2012.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2013.png)

그 결과 나온 computational graph

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2014.png)

Pytorch에서  requires_grad=True인 텐서를 leave nodes에서 찾는다. 그리고 tree search 과정을 거쳐서 필요한 gradient를 모두 계산한다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2015.png)

이렇게 계산한 결과는 w1.grad와 w2.grad에 저장한다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2016.png)

update해주고,(API로는 step()), grad_zero_()로 gradient를 0으로 초기화해준다.

애초에 backward()가 grad를 overwrite하는 게 아니라 더하는 방식이라고 한다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2017.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2018.png)

torch.no_grad()가 있으면 requires_grad=True인 텐서도 더 이상 computational graph에 포함되지 않는다.

안하면 SGD steps사이에 backpropagation이 발생하고, 이에 따라서 memory leakage가 발생한다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2019.png)

...

Python 자체 function을 정의해서 사용하더라도 computational graph에 잘 등록된다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2020.png)

sigmoid에 대한 gradient를 따로 정의를 안하고 위의 graph처럼 계산하면 overflow와 같은 오류가 많이 발생한다.

그래서 아래와 같이 forward와 backward를 함수 혹은 연산 별로 따로 정의한다.

이렇게 정의하면 PyTorch내부에서 auto grad가 된다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2021.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2022.png)

물론 대부분의 경우 그냥 python 정의 function으로도 괜찮다.

## nn Module

---

module에는 layer가 많다.

각 module은 내부에 weight과 bias를 가지고 있다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2023.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2024.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2025.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2026.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2027.png)

## Optimizer

---

torch.optim에 optimizer가 많이 있다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2028.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2029.png)

## Making Own Module

---

__init__()과 forward가 필수로 존재한다.

backward는 구현할 필요 없다. autograd를 이용할 것이기 때문이다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2030.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2031.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2032.png)

## Sequential

---

ResNet처럼 block을 만들어서 중첩할 때 유용하다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2033.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2034.png)

# DataLoader

---

이미 유명한 CIFAR10같은 dataset을 가져올 때 사용한다. 

만약 custom dataset을 사용한다면 dataset class를 스스로 만드는 것이 좋다. 이에 대해서는 reference 참고

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2035.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2036.png)

# Pre-trained Models

---

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2037.png)

# Dynamic Computation Graphs

---

forward할 때마다 computational graph 새로 만들고, backward하고 나면 그 graph를 없애버린다.

장점 : 이거 덕분에 python의 contorl flow를 사용할 수 있게 되었다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2038.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2039.png)

# Alt : Static Computation Graphs

---

1. Build computational graph + find paths for backprop
2. Reuse the same graph on every iteration

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2040.png)

# Alt : Static Graphs with JIT

---

python function으로 쓰면 알아서 graph로 변환해준다고 한다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2041.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2042.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2043.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2044.png)

데코레이터를 사용하면 더 간결하게 사용할 수 있다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2045.png)

# Static vs Dynamic Graphs

---

## Optimization

---

두 layer를 합쳐서 optimize를 용이하게 할 수 있다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2046.png)

Static graph는 python에서 코드를 쓰고 다른 언어로 옮기는 것이 용이하다.

Dynamic graph의 경우 항상 python interpreter가 필요하다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2047.png)

Static graph는 error message가 복잡하다.

Dynamic graph는 디버깅이 쉽다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2048.png)

직전 입력에 영향을 많이 받는 model은 dynamic graph가 되도록 구성하는 것이 좋다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2049.png)

# Tensorflow

---

다른 코드를 볼 떄 유의하기 위해, graph 유형이 버전마다 다름을 인지하자.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2050.png)

## Tensorflow 1.x

---

이건 version 1.0이다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2051.png)

## Tensorflow 2.x

---

이건 version2.0이다. 

require_grad=True를 TF에서는 Variable 변수로 선언해서 처리한다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2052.png)

GradientTape() as tape: 이하로 나오는 것은 전부 Computational Graph를 만드는 것으로 가정한다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2053.png)

tape를 이용해서 바로 gradient를 구할 수 있다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2054.png)

물론 update는 따로 해줘야한다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2055.png)

## Tensorflow 2.0 Static Graph

---

데코레이터를 올림으로써 일반적인 python function으로 Static graph를 만들 수 있다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2056.png)

PyTorch와는 다르게 grad와 update가 static graph와 함께할 수 있다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2057.png)

이제 step function 한 번 으로 grad descent와 update 모두를 해낼 수 있다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2058.png)

## Tensorflow with Keras

---

Pytorch의 Module과 비슷한 기능을 제공한다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2059.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2060.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2061.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2062.png)

혹은 모델을 만들고 opt.minimize를 이용해서 auto grad와 update를 모두 진행할 수도 있다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2063.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2064.png)

# TensorBoard

---

모델의 상태를 알아볼 때 자주 사용한다.

Computational Graph도 볼 수 있다.

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2065.png)

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2066.png)

# 최종적으로,,,

---

![Untitled](Lecture%209%20Hardware%20and%20Software%20a597ed76cb484a0cba5af36475ba7c02/Untitled%2067.png)

# REFERENCE

---

[07. 커스텀 데이터셋(Custom Dataset) - PyTorch로 시작하는 딥 러닝 입문](https://www.notion.so/07-Custom-Dataset-PyTorch-ac0c5a2ddf2249aca7272977f7b54794)

[[Pytorch] 진짜 커스텀 데이터셋 만들기, 몇 가지 팁 :: 취미생활하는 공대생](https://www.notion.so/Pytorch-53323afd09f4438db08f151f72a63c2d)

[[PyTorch] 암 이미지로 커스텀 데이터셋 만들기(creating custom dataset for cancer images)](https://www.notion.so/PyTorch-creating-custom-dataset-for-cancer-images-fea5885549c74bf9a7ec42be53feaf66)