# Lecture 6 : Backpropagation = Auto gradient

# Graph

---

편미분 계산을 위해 Computational Graph를 사용한다.

![Image](https://i.imgur.com/ITau0xb.jpg)

다음과 같은 computational graph가 있다고 하자.

![Image](https://i.imgur.com/7Cifa4n.jpg)

forward pass와 backward pass를 거친다.

forward pass에서는 Loss function까지 쭉 값을 대입해준다.

![Image](https://i.imgur.com/LBuqChp.jpg)

backward pass에서는 차례로 하나의 node를 기준으로만 output과 input의 gradient를 계산하고, input node 쪽으로 그 gradient를 곱해 Chain Rule을 적용해준다.

![Image](https://i.imgur.com/FgCajaW.jpg)

이런 순서로 backward pass가 진행된다.

참고로 아래 사진과 같이 각각의 항을

Downstream Gradient

Local Gradient

Upstream Gradient

라고 한다.

![Image](https://i.imgur.com/FVJewCO.jpg)

![Image](https://i.imgur.com/fSsz9K8.jpg)

하나의 node에 대해서 다음과 같은 process를 여러번 거쳐서, gradient 과정을 modular하게 만든다.

우선 연산 f에 대해서 그 output인 z에 대해서 더 뒤에 있는 output인 Loss에 대한 gradient를 구한다. 그 결과가 바로 $\frac{\partial L}{\partial z}$이다.

그리고 output z에 대해 input x,  y의 gradient를 구한다. 이 둘이 각각 $\frac{\partial z}{\partial x}$, $\frac{\partial z}{\partial y}$이다.

그리고 Chain Rule에 따라서 Loss에 대한 gradient를 구한다.

이 과정을 거치면 $\frac{\partial L}{\partial x}$, $\frac{\partial L}{\partial y}$을 구할 수 있다.

![Image](https://i.imgur.com/qiLI1oX.jpg)

예시로 sigmoid 함수에 대한 backpropagation을 한다고 하면 그 process는 다음과 같다.

![Image](https://i.imgur.com/rDRH6UB.jpg)

![Image](https://i.imgur.com/cAx92t6.jpg)

![Image](https://i.imgur.com/5nAA8EE.jpg)

미분 결과를 아는 module이 있다면 그 내부에서 요구하는 연산 과정은 무시하고 바로 미분 결과를 사용해서 backpropagation을 진행할 수 있다.

![Image](https://i.imgur.com/OvLWZ0H.jpg)

Gradient node를 일종의 gate로 볼 수도 있다. 즉, 연산에 따라 forward/backward를 설정해줄 수 있다는 얘기다.

![Image](https://i.imgur.com/7IqU8Z7.jpg)

Forward pass에서 사용한 코드가 Backward pass에서 사용된다. 영향을 미친다.

참고로 이런 방식으로 backpropagation을 진행하는 것을 Flat하다고 언급한다.

![Image](https://i.imgur.com/WPVfbYR.jpg)

근데 매번 이렇게 코드 짜는 건 귀찮은지 아예 모듈 형태로 API를 제공한다. 이를 사용하자.

![Image](https://i.imgur.com/b3Xnjwy.jpg)

아래는 python으로 구현한 약간의 pseudo code

![Image](https://i.imgur.com/B9Ns8rG.jpg)

간단히 예를 들면 Pytorch에서 Sigmoid의 foward/backward는 다음과 같이 계산된다.

```python
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/Sigmoid.c"
#else

void THNN_(Sigmoid_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  THTensor_(sigmoid)(output, input); # sigmoid output
}

void THNN_(Sigmoid_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output)
{
  THNN_CHECK_NELEMENT(output, gradOutput);
  THTensor_(resizeAs)(gradInput, output);
  TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, gradOutput, scalar_t, output,
    scalar_t z = *output_data;
    *gradInput_data = *gradOutput_data * (1. - z) * z; # gradient of sigmoid
  );
}

#endif
```

# Multivariable Backpropagation

---

이제 matrix로 확장해서 생각해보자. 

각각의 상황에 대해서 그 미분 결과가 Scalar, Gradient, Jacobian이 된다.

각각의 정의와 특성에 대해서는 아래 Own Question을 참고

![Image](https://i.imgur.com/DGTjOo6.jpg)

아래 부분에 써진 것이 각 미분의 물리적 의미를 뜻한다.

Loss는 항상 그 결과가 스칼라임을 명심하라.

## Vector → Gradient

내부연산은 이제  Gradient가 행렬벡터곱으로 표현된다.

![Image](https://i.imgur.com/bTyTohk.jpg)

예시를 들자면 다음 ReLU fucntion에 대한 backpropagation은 다음 그림처럼 행해져야할 것이다.

Jacobian matrix을 보면, 해당 위치의 input만이 그 위치의 output에 영향을 준다는 것으르알 수 있다.

![Image](https://i.imgur.com/8kVY3gW.jpg)

하지만 jacobian을 그대로 메모리에 넣기에는 메모리가 너무 작다! 따라서 다른 implicit multiplication을 사용한다.

![Image](https://i.imgur.com/8fBuLvl.jpg)

## Matrix → Jacobian

![Image](https://i.imgur.com/7h9CBKd.jpg)

여기까지 오면 메모리가 부족해져서 implicit multiplication이 강제된다.

![Image](https://i.imgur.com/rGIGOzR.jpg)

예시를 들자면 다음 product fucntion에 대한 backpropagation은 다음 그림처럼 행해져야 할 것이다.

마찬가지로 Jacobian matrix을 보면, 해당 위치의 input만이 그 위치의 output에 영향을 준다는 것으르알 수 있다.

![Image](https://i.imgur.com/fW33uMd.jpg)

일단 Jacobian을 먼저 구하자. 다음 flow를 따른다.

![Image](https://i.imgur.com/9Qu8x6M.jpg)

이제 Jacobian을 구했으면 Loss에 대한 input의 gradient를 구하자.


![Image](https://i.imgur.com/QCUMXR9.jpg)

![Image](https://i.imgur.com/ihymnI8.jpg)

잘 보면, 기본적으로 다음 공식을 따르게 된다는 것을 알 수 있다.

![Image](https://i.imgur.com/hAxTCDj.jpg)

이를 정리하면, **input에 대한 gradient를 다음과 같이 계산할 수 있다.**

![Image](https://i.imgur.com/FO7ZgtJ.jpg)

앞서 본 gradient를 구하는 방식은 Right to left 곱을 함으로써 backpropagation을 구현한다.

이를 통해 행렬*행렬이 아니라, 행렬*벡터만 한다.

참고로 이 경우는 vector input, scalar output인 경우를 의미한다.

![Image](https://i.imgur.com/ZP1uc0v.jpg)

만약 input이 행렬이 아니라 스칼라라면(그리고 output은 vector라면) 다음 과 같이 foward process를 거쳐야한다.

그래야 행렬*벡터 형태의 연산을 유지할 수 있다.

다만 이런 방식의 계산은 DeepLearning Framework에서는 구현되어있지 않다. 즉, 할 일은 없다는 소리.

![Image](https://i.imgur.com/XDf8cKC.jpg)

고차 미분을 하는 방법

다음과 같이 scalar 형태가 되도록 $\frac{\partial L}{\partial x_0} \cdot v$를 구한 다음 이를 대상으로 backpropagation을 행한다.

![Image](https://i.imgur.com/0oaIqEc.jpg)

참고로 Hessian의 물리적 의미는 x1의 변화에 따른 L의 변화량, x1의 변화에 따른 gradient 변화 속도

Hessian은 Regularization등에서 활용된다.

![Image](https://i.imgur.com/Ns8yF9Q.jpg)

# Own Question

---

## Gradient

---

$\nabla f = (\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2},..., \frac{\partial f}{\partial x_n})$

gradient(그레디언트)는 위 식과 같이 각 변수로의 일차 편미분 값으로 구성되는 벡터입니다. 그리고 이 벡터는 f의 값이 가장 가파르게 증가하는 방향을 나타냅니다. 또한 벡터의 크기는 그 증가의 가파른 정도(기울기)를 나타냅니다.

![Image](https://i.imgur.com/1KME7Kt.jpg)

$f = x^2 + y^2$에서, (1,1)에서 그라디언트는 (2,2)가 된다.

$\nabla f = (\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2},..., \frac{\partial f}{\partial x_n})$

## Jacobian

---

Jacobian(야코비언)은 어떤 다변수 벡터함수(vector-valued function of multiple variables)에 대한 일차 미분(first derivative)으로 볼 수 있습니다.

그레디언트는 다변수 스칼라 함수(scalar-valued function of multiple variables)에 대한 일차 미분인 반면 Jacobian(야코비언)은 다변수 벡터 함수(vector-valued function of multiple variables)에 대한 일차미분입니다. 즉, 그레디언트는 통상적인 일변수 함수의 일차미분을 다변수 함수로 확장한 것이고, Jacobian(야코비언)은 이를 다시 다변수 벡터함수로 확장한 것입니다.

$J = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n}\\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}$

자코비안 행렬은 미소 변화에 관한 선형 변환이라는 것을 알 수 있다.

더 엄밀히는 자코비안 행렬은 미소 영역에서 **‘비선형 변환’을 ‘선형 변환으로 근사’** 시킨 것이다.

![Image](https://i.imgur.com/gA4kwGM.jpg)

이렇게 국소적으로 파악함과 동시에, 국소 지점을 원점으로 잡으면 이를 선형변환으로 볼 수 있다는 논리.

기본적으로 Jacobian은 비선형변환의 과정에서 발생한다.

어떤 선형변환 J에 의해서 du와 dv는 dx와 dy로 변환된다고 볼 수 있다.

![Image](https://i.imgur.com/flP3K3R.jpg)

$\begin{bmatrix} dx \\ dy \end{bmatrix} = J \begin{bmatrix}du \\ dv\end{bmatrix} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} du \\ dv \end{bmatrix}$

$dx = a\times du + b\times dv, dy = c \times du + d \times dv$이

Chain Rule을 이용해서 이걸 잘 정리하면 다음과 같이 Jacobian 행렬을 구할 수 있다.

$J = \begin{bmatrix} \frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} \\ \frac{\partial y}{\partial u} & \frac{\partial y}{\partial v} \end{bmatrix}$

### Determinant of Jacobian

---

행렬식은 선형변환 할 때 단위 면적이 얼마만큼 늘어나는가를 나타낸다. 따라서, Jacobian의 행렬의 행렬식의 의미는 원래 좌표계에서 변환된 좌표계로 변환될 때의 넓이의 변화 비율을 말해준다.

![Image](https://i.imgur.com/OzgmdzO.jpg)

![Image](https://i.imgur.com/nLrhcnS.jpg)

## Hessian

---

$H(f) = 

\begin{bmatrix}
 \frac{\partial^2f}{\partial x_1^2} & \frac{\partial^2f}{\partial x_1\partial x_2} & \cdots & \frac{\partial^2f}{\partial x_1\partial x_n}  \\\\
 \frac{\partial^2f}{\partial x_2\partial x_1} & \frac{\partial^2f}{\partial x_2^2} & \cdots & \vdots  \\\\
 \vdots & \vdots & \ddots & \vdots  \\\\
 \frac{\partial^2f}{\partial x_n\partial x_1} & \cdots & \cdots &  \frac{\partial^2f}{\partial x_n^2}
 \end{bmatrix}$

Hessian 행렬은 함수의 Bowl 형태가 얼마나 변형되었는가를 나타내주고 있다는 것

참고로, 편미분은 그 미분 순서가 중요하지 않으므로 Hessian은 Symmetric Matrix가된다.

**헤시안 행렬의 고유값(eigenvalue)이**

**모두 양수면 => 극소점**

**모두 음수면 => 극대점**

**양수와 음수가 동시에 있으면 => 안장점**

모든 행렬은 선형 변환이고, 선형 변환을 기하학적으로 생각하면 선형 변환이란 일종의 공간 변형이다.

**기하학적으로 Hessian matrix가 시행하고 있는 선형 변환은 기본 bowl 형태의 함수를 좀 더 볼록하거나 오목하게 만드는 변환이다.**

고유벡터와 고유값은 각각 선형변환을 했을 때 그 크기는 변하나 방향은 변하지 않는 벡터와 선형변환 후에 얼마만큼 그 벡터가 변했는지를 나타낸다고 언급한 바가 있다.

Hessian 행렬의 **고유벡터(eigenvector)는 함수의 곡률이 큰 방향벡터**를 나타내고 **고유값(eigenvalue)은 해당 고유벡터(eigenvector) 방향으로의 함수의 곡률(curvature, 이차미분값)**을 나타낸다

## Lapliacian

---

$\nabla^2 f = (\frac{\partial^2 f}{\partial x_1^2}, \frac{\partial^2 f}{\partial x_2^2},..., \frac{\partial^2 f}{\partial x_n^2})$

물리적 의미는 나도 몰?루

Laplacian은 영상의 밝기 변화가 평면형(planar)을 이룰 때 최소의 절대값을 가지고 극대, 극소점처럼 모든 방향으로 밝기 변화가 심할 때 최대의 절대값을 가집니다. 따라서, Laplacian은 영상에서 blob을 찾거나 코너점(corner point)를 찾는 용도로 활용될 수 있습니다.

# Reference

---

[[선형대수학] 헤시안(Hessian) 행렬과 극소점, 극대점, 안장점 by bskyvision](https://bskyvision.com/661)

[헤세 행렬(Hessian Matrix)의 기하학적 의미 - 공돌이의 수학정리노트 (angeloyeo.github.io)](https://angeloyeo.github.io/2020/06/17/Hessian.html)

[다크 프로그래머 :: Gradient, Jacobian 행렬, Hessian 행렬, Laplacian (tistory.com)](https://darkpgmr.tistory.com/132)

[자코비안(Jacobian) 행렬의 기하학적 의미 - 공돌이의 수학정리노트 (angeloyeo.github.io)](https://angeloyeo.github.io/2020/07/24/Jacobian.html)