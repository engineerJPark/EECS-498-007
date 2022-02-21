# Lecture 12 : Recurrent Network

# Intro

---

이제는 새로운 문제를 정의해서 풀어보자.

그동안 풀었던 문제는 feed forward network이다.

feed forward network = one input → one output

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled.png)

여기서 벗어나서 다양한 sequence 형태의 input output을 다루고 싶다.

![이미지를 설명하는 글을 쓸 수 있다.](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%201.png)

이미지를 설명하는 글을 쓸 수 있다.

![비디오에 레이블을 붙일 수 있다.](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%202.png)

비디오에 레이블을 붙일 수 있다.

![번역을 할 수 있다.](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%203.png)

번역을 할 수 있다.

![비디오의 프레임마다 레이블을 붙일 수 있다.](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%204.png)

비디오의 프레임마다 레이블을 붙일 수 있다.

이렇게 sequence 형태의 input ouput을 다루려면 Recurrent Network가 적합하다.

이 문제의 중요한 점은, input output의 sequence length를 모른다는 점이다.

**즉, 우리는 여러 sequence length에 대응할 수 있는 모델을 만들고 싶다.**

# Intro : Sequential Processing of Non-Sequential Data

---

놀랍게도 Non Sequential Data에 대응하는 데에도 좋다.

이렇게 multiple glimpses를 가지고 sequence를 찾는 것.

즉 직전의 step에 영향을 받는다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%205.png)

Generating을 하는데에도 쓰인다.

직전의 generating이 현시점 generating에 영향을 미친다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%206.png)

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%207.png)

# Recurrent Neural Network

---

input으로 hidden state를 update하고, 이를 통해서 output을 만든다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%208.png)

hidden state를 update하는 공식은 다음과 같다.

**여기서 주의해야할 것은 $f_W$와 weight $W$는 항상 동일해야 한다.**

**이를 통해서 모든 sequence에 같은 가중치가 가해지게 된다.**

→ arbitary한 길이의 sequence에 대응할 수 있게 된다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%209.png)

가장 기본적인 RNN의 hidden state eq는 다음과 같다.

sequence의 모든 부분에 대해서 tanh와 $W_{hh}$와 $W_{xh}$는 고정되어야 한다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2010.png)

## RNN Computational Graph

매번 같은 Weight를 사용하므로, weight는 하나의 node에만 있어야 한다.

여기서 Weight node는 Copy 기능을 하므로, backward pass에서 gradient를 sum해야 한다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2011.png)

위 부분까지 작성하면 다음과 같다.

output이 나왔으면 하는 시점을 따로 지정할 수 있다.

many to many는 다음과 같다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2012.png)

many to one

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2013.png)

one to many

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2014.png)

## Sequence to Sequence (Seq2Seq)

번역 등에 쓰인다.

input output 모두 sequence로 나온다. 두 sequence는 length가 다를 수 있다.

Computational Graph는 다음과 같다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2015.png)

왜 두가지로 RNN을 분리했는가?

output seq의 길이가 얼마나 되는 지 모르니까, 따로 구분을 해놓은 것!

물론 하나의 weight만 쓰면서 K개 input K개 output을 하는 경우도 있기는 하다.

## Example : Language Modeling

다음 문자가 무엇인지 맞추는 모델

우선, 각 문자를 one hot vector로 바꾼다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2016.png)

이런 구조로 구성되어있고, output에 대해서 softmax를 먹이고 cross entropy loss 처리를 해서 다음 문자를 예측한다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2017.png)

다음과 같은 순서로 한 문자씩 예측해나간다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2018.png)

이제는 아예 새로운 text를 쓰고 싶은데,

initial seed token에 대해서 어떻게 반응할 지 정하고 싶다.

output으로 나온 예측 결과를 다음 input으로 넣는다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2019.png)

방금 전까지는 input을 one hot vector로 encoding 했다.

보면 그냥 간단하게 하나의 column만 나오기 때문에, **Matrix Multiplying을 하지 않고 Embedding으로 처리할 수 있다.**

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2020.png)

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2021.png)

즉, 가중치를 학습시키는 것은 곧 이 embedding layer를 학습시키는 것과 같은 결과를 불러온다.

...

## BackPropagation Through Time

이렇게 있는 그대로 Computational Graph를 이용해서 BackPropagation을 하면 메모리가 너무 많이 든다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2022.png)

대신 이 방법을 쓴다.

## Truncated Backpropagation Through Time

전체의 sequence를 approximate하겠다는 것.

10~100개의 sequence에 대해서 하나의 chunk로 만들고, 그에 대한 loss를 계산하고 backpropagate한다. 이 때 update도 같이 한다.

그리고 first chunk에서의 hidden weight을 기록했다가 다음 chunk로 넘겨준다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2023.png)

first chunk에서의 hidden weight를 second chunk에서 받는다.

이번에도 마찬가지로 second chunk에 대해서만 loss를 계산하고 backpropagate한다. 이 때 update도 같이 한다.

hidden weight을 기록했다가 다음 chunk로 넘겨준다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2024.png)

같은 방법을 시행한다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2025.png)

forward pass에서는 hidden state를 통해서 sequence를 계속해서 가지고 있다.

하지만 backward pass에서는 중간중간 계속해서 계산이 잘리는 일이 발생한다.

따라서 기억해야하는 데이터의 양이 줄어들어서 적은 메모리를 사용하게 된다.

### Example : Shakespeare 소설을 통한 실험

다음 문제를 예측하는 알고리즘을 그대로 사용했다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2026.png)

아래는 실험 결과

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2027.png)

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2028.png)

이렇게 character 단위로 학습 시키면, 어떤 것이든 학습의 대상이 될 수 있다.

# Interpretablility of Hidden Units

---

RNN이 학습하고 있는 data는 대체 무엇인가??? 이를 알아보기 위해서 hidden unit을 살펴본다.

실험을 위해서, RNN을 unroll 한 후, 많은 step을 학습 시킨다. 그리고 다음 character를 predict하도록 한다.

그 와중에 hidden state를 계속 생성하게 한다.



방금 질문에서, hidden state가 가지고 있는 dimension이 output과 뭐가 다르냐가 중요하다. hidden state가 56이라고 하면, tanh를 통과하면 56개의 -1 to 1인 숫자가 생긴다. 그리고 이 숫자를 이용해서 RNN이 생성 중이던 text에 색칠을 할 것이다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2029.png)

hidden state의 값을 이용해서 RNN이 processing하는 text에 색을 칠했다.

**빨간색은 1에 가깝다는 거고, 파란색은 -1에 가깝다는 의미이다.**

이 경우는 해석할 수 없다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2030.png)

이 경우는 quote를 학습했다고 볼 수 있다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2031.png)

엔터 근처를 학습했다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2032.png)

if 내부의 statement를 학습했다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2033.png)

quote와 comment를 학습

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2034.png)

indentation level을 학습했다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2035.png)

**즉, 그저 다음 character를 예측하는 모델임에도 불구하고, hidden state는 text의 구조를 파악하는 것을 학습했음을 알 수 있다.**

# Example : Image Captioning

---

이미지를 주면 그에 따른 text를 뱉는 모델이다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2036.png)

## 대략적인 순서?

먼저 전이학습을 시키고, 마지막 Conv Layer는 제거한다. 그리고 그 결과는 hidden layer로 넣는다.

첫 번째 input은 start신호를 넣는다. (finite data는 이렇게 하고, infinite data는 이런 거 안한다.)

마찬가지로, 마지막 신호에 end 신호가 나오도록 예측을 하게 만든다. (start token, end token)

공식도 바꾼다. **Convolution에서 data를 hidden state로 넘기기 위해서 $W_{ih} * v$를 추가한다.**

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2037.png)

방금처럼 똑같이 output을 다시 input으로 넣는 것을 반복한다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2038.png)

끝나는 걸 알리는 End token이 마지막에 나와야한다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2039.png)

**여기서 $W_{ih}$는 전이 학습된 Conv Net의 두 번째 마지막 FC Layer에서 오고, v는 입력된 이미지를 한 번 ConvNet을 거친 것에서 온다. 그리고 이걸 각 time step마다 hidden state에 넣는다.**

# Vanilla RNN Gradient Flow

---

일단 gradient가 tanh을 거친다는 문제가 있다

그런데 이건 이것보다 나은 모델이 없어서 해결 불가

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2040.png)

진짜 문제는 매번 gradient에 weight로 곱해지는 것. (+ tanh의 반복 대입)

이러면 exploding 혹은 vanishing이 일어날 수도 있다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2041.png)

그래서 threshold를 정해서 범위 이내로 놓기도 한다. 이를 gradient clipping라고 한다.

하지만 vanishing이 일어나는 경우, 그냥 RNN 구조를 개선해야 한다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2042.png)

# LSTM = Long Short Term Memory

---

Hidden state 하나만 숨기는 게 아니라, 두개의 hidden vector를 만든다

이 두 hidden vector를 각각

cell state

hidden state

라고 이름 붙인다.

cell state는 internal hidden state이다.

그냥 hidden state가 output에 직접적으로 영향을 주는 hidden vector이다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2043.png)

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2044.png)

다음과 같이 W와 x,h의 stack으로 곱하고, 그 결과를 각각 sigmoid 3 stack과 tanh 1 stack에 통과시킨다.

이렇게 나온 결과를 각각

input gate : cell에 어떻게 쓸지 정한다. (0 ~ 1)

forget gate : 기존 cell을 얼마나 지울지 정한다.

output gate : 얼마나 cell을 output으로 내보낼 것인가

g gate : cell에 얼마나 쓸지 정한다. (-1 ~ 1)

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2045.png)

$c_t$와 $c_{t-1}$ 사이의 gradient flow를 보자.

아다마르 곱(각 원소끼리 곱하는 것)은 그냥 백프롭할 때 각 원소별로 그라디언트 곱하는 걸로 처리된다.

여기서 forget gate가 시그모이드를 거치므로, 0~1인 수가 곱해질 것이다.

결과적으로, $c_t$와 $c_{t-1}$ 사이에 non-linearity나 matrix multiplication이 없다.

즉, 방해가 없다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2046.png)

$c_t$와 $c_{t-1}$ 사이의 가장 윗부분에 highway 형성됨.

이걸로만 gradient descent를 한다. Weight에 대해서는 gradient를 하지 않는다.

**(다시 확인할 것)**

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2047.png)

MultiLayer RNN도 가능하다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2048.png)

GRU나 MUT 등의 다양한 RNN 구조도 있다.

GRU나 MUT 모두  add하는 방식으로 gradient flow를 유지한다는 것을 알 수 있다.

MUT의 경우 brute force 방식으로 update formula를 찾아서 적용한다.

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2049.png)

이제는 한 RNN cell이 다른 RNN cell를 추론한다

![Untitled](Lecture%2012%20Recurrent%20Network%20b1d2dbe5839c480e8043da0ecc62838c/Untitled%2050.png)

# Summary

---

RNN으로 네트워크 구조 확장이 가능하다.

Vanilla RNN은 잘 안쓰고, LSTM과 GRU가 자주 쓰인다.

gradient 문제는 다음과 같이 해결한다.

    exploding → gradient clipping

    vanishing → LSTM (additive interaction)

better/simpler architecture are a hot topic

better understanding is needed