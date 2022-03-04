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


