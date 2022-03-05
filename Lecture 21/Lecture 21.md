# Lecture 21 : Reinforcement Learning

- 이번에는 Reinforcement Learning을 배운다.

- Agent, Agent, Environment, Reward로 모델링을 한다.

- 목적은 Reward를 최대로 한다.

![](https://i.imgur.com/4vlR9RH.png)

- Agent : 우리가 제어하는 것

- Environment : Agent에게 reward를 주는 시스템

- 학습 순서
  
  - Agent는 state를 관측한다.  
  
  - state는 noisy하거나 incomplete할 수도 있다
  
  - state를 관측 후, action을 결정한다 
  
  - 그리고 reward를 받는다. Reward는 agent가 얼마나 잘했는지 평가하는 지표이다. Loss function과 같은 목적 
  
  - Action 이후엔 environment도 변경된다. Internal model of the world change 
  
  - 그리고 agent가 reward를 최대로 하기 위한 정책도 변경된다

![](https://i.imgur.com/jV3a6YW.png)

![](https://i.imgur.com/vyTO1YP.png)

# Example

- 각종 예시
  
  - 바둑의 경우 reward와 action까지의 거리가 상당히 멀다
  
  - 즉, reward가 즉각적이지 않아도 된다는 뜻

![](https://i.imgur.com/N3s7ecH.png)

![](https://i.imgur.com/XM4ENrV.png)

![](https://i.imgur.com/t7b51D6.png)

![](https://i.imgur.com/BLoaYOk.png)

# RL vs SL

RL과 SL을 비슷하게 생각할 수도 있지만, 여러가지 차이가 있다.

- RL과 SL의 차이점
  
  - Stochasticity : state, rewards와 state transition이 랜덤할 수 있다.
    - Supervised Learning에서는 Loss가 하나의 함수 형태로 고정적이다.
  - Credit assignment: Reward $r_t$가 action $a_t$에 즉각적이지 않을 수 있다.
    - 따라서 어떤 action이 reward를 유발했는지 알 수가 없다.
  - Nondifferentiable: World를 통해서 Backpropagation을 할 수 없다. $\frac{dr_t}{da_t}$를 구할 수 없다.
    - 어떻게 World가 행동하는 지 알 수 없기 때문이다.
  - Nonstationary: agent가 어떻게 행동하느냐에 따라서 agent가 경험하는 state가 달라진다.
    - Agent가 개선되면, world에서 새로운 action을 하게 된다. 즉, world의 new parts를 explore한다.
    - agent가 학습하는 function은 그 시점에서 agent가 가장 잘 행동하는 방법을 의미한다.
    - 모델이 학습하는 것은 결국 non stationary distribution이 된다.
    - 이는 GAN에서 Generator/Discriminator Net도 마찬가지이다.

- 가능하면 문제를 Supervised Learning으로 정의하는 것이 좋다.

![](https://i.imgur.com/uOkp7zG.png)

# Markov Decision Process (MDP)

- 아래 필기와 같은 구성요소로 Reinforcement Learning이 구성되어 있다.

## Markov Property

- 현재 시점의 state가 현재 시점의 world의 모든 것을 표현한다.

- 전체 시점이 아니라, 오직 직전의 시점만이 현재 시점의 state와 reward에 영향을 준다.

- environmetn와 interact하는 agent를 학습하고, environment는 markov property로 modeling한다.

![](https://i.imgur.com/dnr3fJ5.png)

- agent는 policy를 학습힌다.
  
  - policy $\pi$ : state에 따라서 결정되는 distribution of action

- Goal: cumulative discounted reward: $\Sigma_t \gamma^tr_t$를 최대로 하는 policy $\pi$를 찾는다.
  
  - 하지만 time 0에서의 reward와 time 100에서의 reward를 어떻게 비교하는가?
  
  - 이는 gamma를 이용해서 잘 trade off한다.
    
    - gamma = 0 : 당장
    
    - gamma = 1 : 오직 미래

## MDP in time step

- MDP의 운용 순서

![](https://i.imgur.com/njpAWzM.png)

## MDP Example

- 한 칸씩 옮겼을 때, 가장 높은 Reward를 받는 경로를 구하는 문제이다.

- 시작점은 따로 없다.

![](https://i.imgur.com/iLzb8Ea.png)

- Goal보다 낮은 위치면 위로 가도록하고, 높은 위치면 아래로 가도록 한다

- 양옆으로 갈 필요가 있을 때에는 그렇게 하도록 한다.

![](https://i.imgur.com/sirMysE.png)

# Finding Optimal Policies

- Policy $\pi^*$를 찾는 것이 목적이다.
  
  - Policy $\pi^*$ = discounted sum of rewards를 최대로 하는 정책. Optimal policy = 최선의 정책

- 하지만 initial state, transition probability, rewards 등이 random하다는 문제가 있다.

- 그래서 이를 평균을 내서 그 평균값을 최대로 만드는 것을 목적으로 바꾼다.
  
  - state나 action은 여전히 random하지만 optimal policy $\pi^*$에 dependent하다.

![](https://i.imgur.com/dXC8mKE.png)

# Value Function and Q Function

- policy $\pi$는 sample trajectory(path)를 만든다.
  
  - $s_0, a_0, r_0$ ... $s_1, a_1, r_1$ ...

- 각 state에 얼마나 잘하고 있는 지 측정하고 싶다.

- Value Function
  
  - 최적 State를 찾는다.
  
  - policy $\pi$를 적용하고 state s에서 시작한다면 얼마만큼의 reward를 받는지 
  
  - 즉, 해당 s가 얼마나 좋은지 평가하는 것

- Q Function
  
  - 해당 state에서 최적의 action을 찾는다.
  
  - Q Function이 수학적으로 더 편하다.
  
  - 시작점 state s에서 action a를 행한 후, policy $\pi$를 행한다면 얼마만큼의 reward를 받는지
  
  - 즉, 해당 (s, a) pair가 얼마나 좋은지 평가하는 것

![](https://i.imgur.com/MZ9LCGY.png)

- 질문 리스트
  
  - 보통 Value Function 따로, Q Function 따로 다루게 된다.
  
  - Value Function, Q Function은 서로 비슷한 걸 측정하는 것이다.

# Bellaman Equation

- Optimal Q-Function
  
  - state s에서 action a를 행한 후 optimal policy $\pi^*$를 행했을 때의 reward를 의미한다. 
  
  - $Q^*$는 $\pi^*$를 함축한다. 즉, $Q^*는 $ $\pi^*$를 $Q^*$로 표현할 수 있다.
  
  - 따라서 policy를 그냥 Q Function 하나로 퉁쳐서 Q Function만을 신경 쓸 수 있다.

- Bellman Equation
  
  - Recurrence equation
  
  - 같은 의미이다. state s에서 action a를 행한 후 optimal policy $\pi^*$를 행했을 때의 reward를 의미한다.
  
  - 여기서 $s', a'$는 다음 state와 action을 의미한다.

![](https://i.imgur.com/9H171Y6.png)

- 방금 policy를 Q function으로 바꿨으니, 목표를 optimal Q function을 찾는 것으로 바꾼다.

- Bellman Equation을 만족하는 Q function을 찾으면, 이는 무조건 optimal Q function이다.

- Bellman Equation을 iterative update rule로 사용한다.
  
  - 초기에는 random Q function에서 시작하고, Bellman Equation을 이용해서 계속 Q function을 update한다.
  
  - 이걸 반복하면 optimal Q function이 만들어진다.

- 문제점
  
  - 모든 (state, action) pair를 거쳐와야한다. 너무 많은 계산이 필요하다.
    
    - 그래서 Bellman Equation을 Loss로 사용하고, Q(s, a)를 Neural Network를 이용해서 추정한다.

![](https://i.imgur.com/XttKcKO.png)

- parameter $\theta$와 함께 training한다.
  
  - input으로는 s, a, $\theta$
  
  - 예측값이 Bellman Equation을 만족해야하므로, Bellman Equation을 이용해서 해당 state와 action에서의 Q 값을 추정한다.
  
  - Loss를 구한다. network의 output이 Q function을 따라가도록 한다.
    
    - 그 결과 해당 state와 action을 넣어주면 알아서 optimal policy를 구할 수 있는 것이다.

- 문제점
  
  - Nonstationary함이 남아있다. Q(s,a)의 예측값이 weight $\theta$에 의존적이다. **즉, 학습됨에 따라서 달라진다.**
  
  - training data를 sample해서 batch로 만들 명확한 방법이 없다.

![](https://i.imgur.com/IgScqI2.png)

# Case Study : Atari

- 대표적인 예시

![](https://i.imgur.com/xsOmqPf.png)

- 화면 4장을 받는다

- CNN을 거쳐서 Action(Q function)을 output한다

![](https://i.imgur.com/LbZA9gv.png)

# Policy Gradients

- 어떤 Problem에서는 Q function을 학습시키기 어려울 수 있다.

- 어떤 problem에서는 Q function을 거치지 않고 state to action을 바로 학습할 수도 있다.

- Policy gradient
  
  - input **state**, output **distribution of actions at that state**
    
    - Obejective function은 expected sum of future reward로, Maximize하고자 하는 것
    
    - input **weight**, output **expected sum of future reward**
  
  - 이제는 여기에 Gradient Ascent를 한다.

![](https://i.imgur.com/JRrRBr8.png)

- 문제는 이게 미분가능하지 않다는 것이다. Gradient를 environment를 통해서 해야해서

- 이를 해결하기 위해서 좀 더 general한 case에 대해서 gradient를 구해보자.
  
  - x = trajectory of states and actions and rewards
  
  - p(x) = distribution of trajectories (이 trajectory는 policy에 의해 결정된다.)
  
  - f(x) = reward function after x

![](https://i.imgur.com/UcKiAH3.png)

- 결과적으로 reward function과 log probability of trajectory의 평균이 나오게 된다.

- 몇 개의 trajectory를 sampling해서 approximate할 수 있다.

![](https://i.imgur.com/yBNkpJu.png)

- 마저 정리하자.

- logP(x)를 다음과 같이 두가지 term으로 만들 수 있고, 각각 의미는 다음과 같다.
  
  - Transition probability of environment. 계산 불가. environment 전체를 뒤질 수는 없기게,,,
  
  - Action probabiltiy of policy. 이것은 계산할 수 있다. 우리가 이걸 학습 중인 것이니까.

![](https://i.imgur.com/odDXcpN.png)

- 이걸 대입해서 Cost Function의 gradient를 구하면 다음과 같다.

![](https://i.imgur.com/Y4QPLfn.png)

- 각 항목의 의미를 살펴본다.

- trajectory x (by following policy in the environment)
  
  - 그냥 policy를 실행하게 놔두고 sampling하면 모을 수 있다.

![](https://i.imgur.com/qM69DtW.png)

- f(x)는 reward
  
  - 그냥 policy를 실행하게 놔두고 sampling하면 모을 수 있다.

![](https://i.imgur.com/Tj0Bgec.png)

- action Probability를 학습 중이었으니, policy도 gradient를 구할 수 있다.

- gradient of the predicted scores (respect to the weights)

![](https://i.imgur.com/GNu70X9.png)

- 아래 알고리즘을 통해서 policy를 학습할 수 있다.

![](https://i.imgur.com/lgPq8Ys.png)

- reward high -> f(x) high

- reward low -> f(x) low

- 이에 대한 해석은 다음과 같이 한다.

![](https://i.imgur.com/LEyaZBg.png)

- 정리하자면 다음과 같은 방식이 있다
  
  - 여기에 baseline을 추가해서 gradient estimator의 variance를 줄일 수 있다.

![](https://i.imgur.com/QC8atU6.png)

- 그게 바로 이 Model Based Reinforcement Learning이다.

- 유명한 건 Actor Critic과 Model Based

- Actor-Critic
  
  - Actor : predict action at given state
  
  - Critic : How good the state action pair

- Model Based
  
  - state transition을 modeling한다. 그리고 그 model을 학습

- Supervised Learning
  
  - Imitation Learning
  
  - Inverse Reinforcement Learning
  
  - Adversarial Learning

![](https://i.imgur.com/X44n6xF.png)

- 대표적인 예시로 알파고...라는 것이 있다.

![](https://i.imgur.com/YJLYgxY.png)

# Stochastic Compuatation Graphs

- 이렇게 nondifferentiable한 것을 학습할 수 있다.

- image classification을 하는데, 어떤 CNN을 사용할 지 정하는 모델인 것. 

![](https://i.imgur.com/7lh9PMA.png)

# Stochastic Compuatation Graphs : Attention

- 위의 예시

- Reinforcement Learning을 이용하면 Attention에서 정확히 한 region만을 사용할 수 있다. 

- 이를 Hard Attention이라고 한다.

![](https://i.imgur.com/TSLXSpx.png)

# Summary

![](https://i.imgur.com/3tizQgE.png)
