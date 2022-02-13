# Lecture 11 : Training Neural Network 2

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled.png)

# Learning Rate

---

높은 lr로 시작해서 낮은 lr로 변경해나간다... 이를 통해서 learning rate를 수정하는 번거로움을 방지.

**이런 방법을 weight decay, learning rate scheduling이라고 한다.**

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%201.png)

이렇게 step으로 변경하는 것이 초창기 방식

하지만 몇 번째 trian에서 변경할지, 얼마나 변경할 지 등 hyperparameter가 늘어나서 불편

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%202.png)

그 다음 방식이 cosine 방식. 한 때 유행

hyperparameter가 딱히 없어서 편리하다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%203.png)

그 외에도 다양한 방법이 있다. NLP에서는 Linear 더 많이 쓴다고 한다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%204.png)

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%205.png)

하지만 근본은 Constant다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%206.png)

몇가지 팁에 대해서는

Lrsche는 맨마지막에 갈겨라

Lrsche는 자동으로 하지말고 그래프 보면서 해라

Adam 등은 constant한게 좋고, SGD momentum은 lrsche를 써라...

EARLY STOPING : Validation accuracy가 떨어지는 시점에 대해서 weight를 저장해놓고

그래프를 그린 결과를 보고 어느 시점의 가중치를 쓸지 고르는 거다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%207.png)

# Hyperparameter tuning

---

## Grid Search

사실상 ‘다해보기’ 

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%208.png)

## Random Search

랜덤하게 하기

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%209.png)

이제 두가지를 비교해보자.

랜덤하게 하는 것이 important parameter에 대한 feature를 더 많이 얻을 수 있다

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2010.png)

# 돈이 없는 당신을 위한 Hyperparameter tuning 과정

---

크게 다음과 같은 과정을 거친다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2011.png)

적은 GPU로 어떻게 tuning 해야하는가

## Step 1 : Check initial loss

weight decay와 같은 기능을 모두 끈다.

sanity check를 한다. 얘를 들어 학습 직전의 loss에 대해서 log(C) for softmax with C classes여야하는 것 등.

## Step 2 : Overfit a small sample

5~10개의 minibatch를 learning rate, weight initialization을 킨 상태로 예측률 100%가 나오도록 학습시킨다. 이를 위해서 regularization만 끈다.

이때 Loss에 관해서,

Loss가 떨어지지 않으면, LR이 너무 낮거나, initialization이 나쁜 것.

Loss가 급등하면, LR이 너무 높거나, initialization이 나쁜 것.

이 단계에서 모든 bug를 다 잡아야 한다!

여기까지가 코드의 검증 과정

## Step 3 : Find LR that makes loss go down

이전 단계에서의 architecture 그대로 이용하고, 모든 training dataset을 이용한다.

weight decay를 작은 값으로 ON한다.

100 iteration 이내에서 loss가 크게 떨어지는 learning rate를 찾는다.

보통 learning rate는 1e-1,1e-2,1e-3,1e-4를 테스트한다.

## Step 4 : Coarse grid, train for ~ 1-5 epochs

앞서 성공한 것 중 몇 개만 골라서 대충 grid search를 한다.

1 ~ 5 epoch를 해본다.

이번에는 weight decay도 조금 조정해본다. 

보통 weight decay는 1e-4,1e-5,0를 테스트한다.

## Step 5 : Refine grid, train longer

앞서 좋았던 조합을 골라서 10~20 epoch를 학습시킨다.

## Step 6 : Look at learning curves

learning curve를 본다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2012.png)

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2013.png)

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2014.png)

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2015.png)

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2016.png)

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2017.png)

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2018.png)

## Step 7 : GOTO step 5

결과적으로, 다음과 같은 Hyperparameter를 보통 다루게 된다. 

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2019.png)

Tuning 결과를 모아놓고 비교해라.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2020.png)

Ratio를 살펴라. 본래의 weight Magnitude보다 update보다 크면 이상한 것.

보통은 0.001 ~ 0.01 정도여야 한다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2021.png)

# Model Ensemble

---

각각의 여러 모델을 학습 시키고, 테스트를 그 모든 모델에 한 다음 평균을 낸다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2022.png)

여러 모델 대신에 여러 체크포인트를 사용할 수도 있다.

Periodic하게 learning rate를 조절하는 것이 이 방법에 좋다.

그래야 cost가 낮은 지점 여러 곳을 기록할 수 있기 때문.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2023.png)

running Weight를 저장한다. 그리고 test time에는 Model weight를 가중 평균 내서 사용한다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2024.png)

# Transfer Learning

---

앞서 본 Fully Connected Layer나, Conv Net에서 첫 번째 Fully Connected Layer는 앞선 input의 feature(template)를 학습한다고 했다.

따라서, 만약 deep한 Conv net이 있다면, 앞 부분의 Convolution Layer는 Feature Extracter로 사용하고, 다른 Class를 Classification하는 데 사용할 수 있도록, 마지막의 Fully Connected Layer만 교체해주면 될 것이다.

그렇게 하면 FC Layer가 학습한 Feature template를 기반으로 다른 Class를 잘 Classificate할 수 있게 된다.

이 Transfer Learning을 사용하면 적은 data로도 좋은 결과를 내게 된다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2025.png)

성능 자체도 좋게 나오고,

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2026.png)

다른 method랑 같이 쓰니깐 성능도 오른다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2027.png)

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2028.png)

이렇게 이전의 최고 기록치를 전부 갱신해버렸다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2029.png)

1. 큰 dataset에서 학습한다.
2. Conv Layer의 weight는 고정하고, last FC Layer를 떼어낸 후 적용하는 class에 맞는 숫자로 교체한다.
3. 조금 큰 dataset에 사용하는 경우, **Fine-Tuning**을 한다. 즉, 2번까지 한 후, training을 계속한다는 것!

### Tips

1. FC Layer가 feature extraction을 먼저 학습한 뒤, fine tuning을 실시해라
2. fine tuning할 때, learning rate를 많이 낮춰라
3. 앞 부분 Layer의 weight를 학습 정지 시켜라

...

전이 학습이 잘되는 모델이 따로 있다!

ImageNet에서 잘 되던 모델은 다른 문제에서도 잘 적용된다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2030.png)

new dataset이 ImageNet과 비슷하냐 아니냐에 따라서 다음 4가지 접근 방법을 알 수 있다.

문제는 ImageNet하고 dataset이 많이 다르면서 category당 data 수가 적은 경우. 이 경우는 전이 학습으로 해결이 잘 안난다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2031.png)

오른쪽 위 칸, 빨간색으로 써져있는 대로, pre trained된 linear classifier면 괜찮은 결과가 나올 수 있지만, 여전히 danger zone인 것은 똑같다.

이로써 서로 다른 각기의 문제에 대한 pre training과 fine tuning하는 pipeline을 찾는 것이 중요한 문제가 되었다.

물론 pre training 없이 처음부터 학습해도 잘 된다. 시간이 3배 정도 걸리지만.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2032.png)

결국 transfer learning에서는 pretraining과 finetuning으로 효과를 크게 볼 수 있음을 알 수 있다.

물론 data가 많은 것이 더 좋다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2033.png)

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2034.png)

# Distributed Training

---

다양한 분산 시스템 구성 방법을 알아보자.

## Model Parallelism : Split Model Across GPUs

Idea1 : GPU를 하나만 쓰게 되는 거랑 다를 게 없다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2035.png)

Idea2 : 이것도 어렵다. GPU를 동기화해야한다. 그리고 activation과 grad에 대한 communication process가 필요하다.

참고로 이것이 AlexNet에서 한 방법...

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2036.png)

## Data Parallelism : Copy Model on each GPU, split data

보통의 방법은 이거다.

각각의 batch를 서로 다른 GPU에 넣는 것.

그리고 한 번의 iteration마다 communication을 통해 gradient의 sum, update를 교환한다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2037.png)

K개의 GPU로 늘릴 수도 있다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2038.png)

Model parallelism과 Data Parallism을 합한다.

하지만 이런 거 쓸려면 큰 minibatch를 train하는 일이어야만 한다. 아니면 엄청난 비효율.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2039.png)

# Large-Batch Training

---

많은 GPU를 사용해서 학습 시간을 줄이고 싶다.

minibatch가 커지고, gradient descent step의 개수가 줄어든다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2040.png)

**K개의 GPU를 쓰면, Batch size를 K배하고, 동시에 Learning Rate도 K배해라**

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2041.png)

이 떄 Learning rate 갑자기 키우면 폭발할 수 있으므로, 아래 그래프 모양을 따라서 learning rate를 scheduling한다.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2042.png)

다른 주의할 점.

Batch Norm의 경우, 하나의 GPU 내에서만 normalize할 것.

![Untitled](Lecture%2011%20Training%20Neural%20Network%202%209a1cc13c98e24612996595a1a324e1b4/Untitled%2043.png)

# REFERENCE

---

[정규화 정리 (Regularization) (velog.io)](https://velog.io/@hihajeong/%EC%A0%95%EA%B7%9C%ED%99%94-%EC%A0%95%EB%A6%AC-Regularization)

[[기술면접] L1, L2 regularization, Ridge와 Lasso의 차이점 (201023) (tistory.com)](https://huidea.tistory.com/154)

[[Part Ⅲ. Neural Networks 최적화] 2. Regularization - 라온피플 머신러닝 아카데미 - : 네이버 블로그 (naver.com)](https://m.blog.naver.com/laonple/220527647084)