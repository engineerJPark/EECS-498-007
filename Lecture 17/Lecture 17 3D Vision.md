# Lecture 17: 3D Vision

---

3D problem에 대해서는 크게 두가지 문제에 대해 알아볼 것이다.

하나는 3D shape predicting, 하나는 3D shape classification.

![](https://i.imgur.com/5viz4Qp.png)

그 외에도 다양한 문제가 있다. 

![](https://i.imgur.com/f8tPAQG.png)

3D를 표현하는데 다양한 방법이 있다.

![](https://i.imgur.com/cyzieCs.png)

# Depth Map

---

![](https://i.imgur.com/Ve10rhE.png)

그 중에서도 Depth Map에 대해서 알아 볼 것이다.

각각의 픽셀에 대해서 camera로부터의 거리를 측정한다.

즉, 거리라는 channel이 하나 더 생기는 것이다. 그렇게 RGBD 이미지가 탄생

완전한 3D는 아니므로 2.5D라고 부른다. 완전한 D값을 추출하는 것이 불가능 하기 때문.

하지만 RGBD 이미지는 위 그림처럼 visible portion에 대해서만 D값을 정확히 추출해낸다. 소파 뒤 같은 곳은 안됨

![](https://i.imgur.com/mVnjd9y.png)

일반 카메라를 이용해서 Depth Map을 예측하는 것도 할 수 있다.

앞서 본 Senmentic 문제에서 Fully Convolutional Network와 같은 원리다.

output channel이 하나로 한정되어있고, 그 하나의 channel이 depth의 의미를 띄게 된다.

![](https://i.imgur.com/kCYBSNO.png)

하지만 이것은 불가능하다는 것이 학계의 정설

크고 멀리있는 물건과 작고 가까이 있는 물건의 차이를 밝힐 수가 없다. 

그래서 2D 이미지 하나로는 실제 depth를 측정하기 곤란하다.

![](https://i.imgur.com/PCYDocc.png) 

이 문제를 해결하기 위해서 structure를 변경하기로 했다.

특수한 loss를 사용한다. **scale invariant loss**

prediction이 Ground Truth Depth의 일정한 scale 만큼 예측되도록 하는 것이다.

**모든 prediction이 GT Depth의 1/2만큼이라고 하자.**

**그러면 Scale Invariant Loss는 여전히 0이 된다. 즉, 방향만 따지는 Loss라는 이야기. 이 Loss는 곱해서 GT Depth가 되는 scalar가 단 하나라도 있으면 0이 되는 것이다.**

이 방법을 쓰면 Loss를 0으로 만들면서 실제 Depth에 가까운 prediction을 할 수 있다.

### ※ 참고

시스템이나 함수, 통계 정보에 특정 양만큼 Scale을 해주어도 그들의 성질이나 모형이 변하지 않는다면, 그것은 Scale-Invariant한 시스템, 함수, 통계 정보를 말한다. , Linear system의 성질 중 Homogenity(동질성)과 같은 성질이다. $f(λx)=λΔf(x)$

멀리서 봤는데도 미인인데 가까이서 봐도 역시 미인이면 규모에 불변한다고 하여 Scale invariance라고 합니다.

![](https://i.imgur.com/egMWhUp.png)

## Surface Normal

표면의 normal vector의 방향을 픽셀마다 표시한다.

그리고 이 방향을 또 RGB로 표시한다. 예를들면, 블루는 위 레드는 왼쪽

![](https://i.imgur.com/UjRQm8E.png)

3 Channel Output으로 둔다. x,y,z를 표현하도록

그리고 loss를 특이한 것을 쓴다. 그림의 공식을 보면 두 이미지 사이의 벡터 사이각을 측정하는 것임을 알 수 있다.

![](https://i.imgur.com/wYUx0ry.png)

Fully Convolutional Network로 Segmentation, Surface Normal, Depth Map을 구할 수 있다. 하나의 이미지를 넣고 세가지 출력을 낼 수 있는 것이다.

하지만 여전히 가려진 부분은 3D 표현을 할 수가 없다.

# Voxel Grid

---

![](https://i.imgur.com/aCQVfmM.png)

세상을 3D Grid로 표현하고 각 cell이 있는지 없는지만 표현하는 것.

마치 마인크래프트처럼

Mask RCNN과 같은 메커니즘을 사용할 수 있다.

그러나 아래 그림과 같은 문제점이 있다.

![](C:\Users\jshac\AppData\Roaming\marktext\images\2022-02-16-16-52-46-image.png)

이번에는 이 Voxel Grid를 input으로 해서 Classification을 해보자.

block이 1혹은 0인 곳에서 3D kernel이 sliding하면서 convolution을 하는 것이다

input : $w*h*d*(occupied)$ 

output : $w*h*d*(occupied)$![](https://i.imgur.com/KhCUMKG.png)

이번에는 2D 를 받아서 3D Voxel Grid를 뽑는 Network에 대해서 알아보자.

training을 cross entropy loss로 할 수 있다. 해당 위치에 block이 있는지 없는지 확인하는 것이니까.

보통 Fully Connected Layer를 이용해서 3d ~ 4d tensor 간의 간극을 해결한다.

FC Layer 앞에서 flatten하고, Fully connected layer를 거쳐서 4d Tensor로 변환한다.

그리고 이를 Upsampling하는 것.

하지만 3D convolution은 expensive하다.

![](https://i.imgur.com/BjP2Jow.png)

그래서 사람들은 종종 2d convolution만을 이용해서 voxel grid를 추정하려고 하기도 한다.

voxel tube

마지막 2D Convolution Layer channel이 tube of the voxel score(마지막 channel을 tube of voxel probability로 인식)로 인식되는 것이다.

Loss를 계산할 때, channel을 depth dimension이라고 가정하고 계산하는 것이다. 

참고로 이런식으로 계산하면 transitional invariance in Z direction을 잃는다.

![](https://i.imgur.com/dbZizAs.png)

높은 해상도에서는 사용할 수가 없다. grid 저장에만 4기가바이트가 쓰이니깐

![](https://i.imgur.com/WCKmQWC.png)

일종의 trick

처음에는 작은 해상도로 구현하고, 디테일이 필요한 곳(sparse subset of voxel cell에)마다 더 큰 해상도로 구현하는 방법

구현이 힘들다

    multi resolution

    sparse representation of voxel grid

![](https://i.imgur.com/HLwSHOd.png)

다른 Trick : Nested Shape Layers

겉 부분 만들고, 제거할 부분 만들고 겉부분 만들고, 제거할 부분만들고... 를 반복해서 하나의 3D object를 표현

![](https://i.imgur.com/78yaGXn.png)

# Implicit Surface

---

![](https://i.imgur.com/Qp57Cfa.png)

**3D shape을 표현하는 function을 학습한다.** 확실하게 주어진 function을 이용해서 학습한다.

input : 3 dimensional coordinate

output : probability that occupied or not(inside or outside)

voxel grid는 한 지점에 대해서 sampling하고, 그 지점의 block 존재 유무를 저장하는 방식인 반면, Implicit function은 말 그대로 implicit한 함수를 사용해서 3d shape을 표현한다. 

그리고 이 function을 통해서 sample하면, object의 inside인지 outside인지 판단하다.

여기서 occupancy probility가 1/2인 등고선을 surface로 지정한다.

![](https://i.imgur.com/KudXyDj.png)

3d shape dataset -> classify outside or inside

explicit하게 함수를 얻고 싶다.

그러므로 grid space에서 각 점에 대해서 sampling을 한다.

그럼 function에 의해 inside/outside가 판명이 난다.

그러고나서 다른 부분에 대해서 다시 grid를 만들고 inside outside를 구분하는 과정을 반복한다.

Oct-Trees 처럼 multiscale output이 가능하다.

![](https://i.imgur.com/VLjdNq9.png)

하지만 구현이 어렵다.

    NN의 구조

    SDF로부터 image 정보를 받아오는 것

    3D 형태를 추출하는 정확한 알고리즘

# Point Cloud

---

35:22

![](https://i.imgur.com/ey8B4uG.png)

![](https://i.imgur.com/WMEPmqK.png)

![](https://i.imgur.com/9VGVAwk.png)

![](https://i.imgur.com/omTVZJ9.png)

![](https://i.imgur.com/MTXFknT.png)

# Mesh

---

![](https://i.imgur.com/HzWxJBt.png)
