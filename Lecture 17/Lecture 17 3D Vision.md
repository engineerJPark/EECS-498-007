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

![](https://i.imgur.com/ey8B4uG.png)

Fine한 곳과 아닌곳 따로 point의 개수를 유연히 다룰 수 있다.

Post processing 해야 실물을 볼 수 있다

![](https://i.imgur.com/WMEPmqK.png)

포인트 클라우드 받고 그걸로 분류하는 구조

하지만 포인트 간의 순서를 학습하지 않도록 하야한다

어떻게?

각각의 point(3차원 coordinate)에 대해서 MLP를 적용해서, 각 point에 대한 독립적인 point feature를 추출해낸다.

그리고 이를 Max Pooling, 모든 point의 feature vector를   하나의 vector를 만들고, FC Layer를 한 번 더 거쳐서 Class score로 만든다.

여기서 order에 무관하게 만드는 연산은 max pooling이다. max 함수 자체는 순서를 따지지 않기 때문.

![](https://i.imgur.com/9VGVAwk.png)

이번에는 Point Cloud를 생성하는 알고리즘이다.

![](https://i.imgur.com/omTVZJ9.png)

포인트 클라우드 예측에 관한 새로운 로스 퐁션을 선언한다.  챔퍼 거리는 두 셋이 얼마나 먼지 표현한다

모든 파란 공에서 가장 가까운 주황공을 찾고 유클리드 거리를 측정하고 합한다

두번째 항은 주황색 입장에서 가장 가까운 파란 공을 찾아서 유클리드 거리를 합하는 것이다

로스가 0이 되려면 두 셋이 완전히 일치해야한다

이 로스를 쓰면 점의 순서는 신경쓰지 않아도 된다

![](https://i.imgur.com/MTXFknT.png)

# Mesh

---

매쉬는 점에 삼각형 표면을 만들어놓은 것, 특징은 그대로이다.

![](https://i.imgur.com/HzWxJBt.png)

첫 번째 방법으로는 vertex를 이용해서 triangle face를 만드는 것이다.

![](https://i.imgur.com/qRsCSYg.png)

![](https://i.imgur.com/wGSWBvj.png)

![](https://i.imgur.com/sTQTwVv.png)

flat surface를 만들기도 용이하고, fine graphic을 위해서 surface를 추가하기도 쉽다.

또한 다른 데이터를 vertex마다 추가하고 이를 중간 지점에서는 interpolate해서 사용하기도 쉽다.

다만 Neural Network에서 다루기 쉽지 않은 자료구조라는 문제가 있다.ㄴ

![](https://i.imgur.com/pg2AWJU.png)

## Predicting Mesh : Pixel2Mesh

이번에는 input image를 주면, 그에 맞는 Mesh를 추출해내는 Network를 살펴본다.

![](https://i.imgur.com/Exca0Ly.png)

### Iterative Refinement

타원형 구의 vertex를 조금씩 변환해서 이미지에 근접한 형태로 만드는 것이다.

![](https://i.imgur.com/GTDo7vv.png)

### Graph Convolution

Graph 내부를 sliding하면서 연산하는 새로운 Convolution이다.

원래 Convolution과 동일하게, 중심 위치와 주변의 다른 node들이 동시에 관여된다.

input : Graph, vertex에 각각의 feature vertor가 있다.

![](https://i.imgur.com/7WR9xfL.png)

이 연산을 기반으로 Graph convolution layer를 stacking한다.

매 iteration마다. 각 vertex에 있는 feature vector를 개선한다. Graph Convolution을 이용하면 neighbor의 영향을 받은 채로 개선되게 되는 효과가 있다.

![](https://i.imgur.com/wingsvw.png)

하지만 원래 우리 목적은 2D image -> mesh 였다.

이를 위해서는 2D image를 graph로 바꿔줄 것이 먼저 필요하다.

### Vertex-Aligned Features

각 vertex마다 feature와 spatial position을 모으는 feature vector를 만드는 법을 알아보자.

image를 CNN에서 Feature를 구한다.

weight 역할인 mesh를 각 평면에 project한다.

여기서 project한 위치가 완전한 grid가 아니면 bilinear interpolation을 해서 그 feature를 ㄴ구한다.

![](https://i.imgur.com/bVRQX1Z.png)

RoI Align할 때 나오던 Bilinear interpolation과 비슷하다.

![](https://i.imgur.com/3fh8Klm.png)

### Loss Function

Loss function을 만들자.

문제는 같은 형상을 서로 다른 mesh를 이용해서 표현할 수 있다는 점이다.

![](https://i.imgur.com/ZFpQTum.png)

그냥 mesh 위에 점을 여러 개 찍고, point cloud의 Loss function을 써라. Chamfer distance

![](https://i.imgur.com/W5tBAWi.png)

point를 찍는 시점이 prediction은 예측할 당시이고, GT image의 경우, 미 리 찍어놓고 한다.

![](https://i.imgur.com/RG1J9FK.png)

# 3D Shape prediction

---

![](https://i.imgur.com/AmNVABQ.png)

우선 Metric에 대해서 언급한다.

## Metric

앞서 bounding box로 IoU를 측정했듯이 비슷하게 측정할 수 있지만, 이게 좋은 방법이 아님이 드러났다.

![](https://i.imgur.com/MqnUIAm.png)

![](https://i.imgur.com/OkoNJtC.png)

다른 방법은 Chamfer Distance와 비슷하게 하는 것. 모든 vertices를 분리하고 이를 point cloud 형태로 둔다음에 chamfer distance을 구한다.

하지만 이것도 문제 : L2 distance 기반이라서, outliers에 굉장히 민감해진다.    

![](https://i.imgur.com/vtBVskK.png)

그래서 사용하는 것이

F1 Score이다.

아래는 주황이 predicted, 파랑이 Ground Truth이다.

shpere를 predicted point에 생성하고, 일정 범위 이내로 GT point가 존재하면, Precision은 옳은 것이다.

반대로, shpere를 GT point에 생성하고, 일정 범위 이내로 predicted point가 존재하면, Recall의 값을 올린다.

![](https://i.imgur.com/Uq1lHnY.png)

F1 score는 3D data를 비교하기에 적합한 규격이다.

조금 더 outlier에 둔감하기 때문이다.

![](https://i.imgur.com/7a0FwcB.png)

![](https://i.imgur.com/utuklmr.png)

## Camera Systems

3D data를 다룰 때에는 어떤 coordinate를 사용할 지 결정해야한다.

다음과 같이 구분할 수 있다. 여기서 canonical coordinate는 직교좌표계를 얘기하는 것

![](https://i.imgur.com/MY7cYjX.png)

canonical한 방법이 더 쉽다.

다만 output의 feature가 더 이상은 input의 feature과 정렬되어있지는 않다는 것이다.

하지만 이는 View point coordinates를 사용하면 해결된다.

![](https://i.imgur.com/Rn4NWKr.png)

![](https://i.imgur.com/WCkon14.png)

결론 : **View Coordinate가 더 성능이 좋다.**

만약 View coordinate를 이용한다면, View Centric Voxel Prediction을 할 수 있고, 매우 자연스러운 결과가 나온다.

이는 input image와 aligned된 tube로 predicted 되기 때문이다...

![](https://i.imgur.com/auOuc96.png)

frustums : 입체를 평행한 두 평면으로 절단 할 때 그 두 평면 사이의 부분을 의미한다. 여기서 입체는 보통 원뿔 혹은 각뿔이다.

## Datasets

ShapeNet : CAD 파일, Pix3D : 실제 가구에 3D CAD(mesh)를 적용.

![](https://i.imgur.com/iO7JO1S.png)

# Mesh R-CNN

---

저번에 말하다가 만 것.

이미지를 받아서 3D 파일을 만들고 싶다.

![](https://i.imgur.com/xrsTpSe.png)

![](https://i.imgur.com/MEBpQmw.png)

위상적으로 동일한 것만 표현 가능하다. 즉, 평면으로 도넛처럼 구멍이 뚫려 있는 것을 만들 수는 없다.

아래 그림 두번째 행이 바로 그런 경우, 구멍이 있어야하는데 초기 조건이 구멍이 없어서 저렇게 난잡한 형상이 남아있다.

![](https://i.imgur.com/o2XhP64.png)

그래서 이를 해결하기 위해서 voxel prediction을 먼저 해서 initial mesh prediction을 만들고, 이에 대해서 mesh를 가다듬는 방식으로 진행한다.

![](https://i.imgur.com/S8QMfB9.png)

다음과 같은 절차를 거치게 된다.

![](https://i.imgur.com/fgH5NcL.png)

Chamfer loss만 사용하면 mesh가 degenerate된다. 특히 edge가 잘 안잡힌다. 그래서 mesh regularizer를 추가해서 사용한다.

![](https://i.imgur.com/31h76fT.png)

Amodal Dectection의 기능을 가지게 된다 =  가려진 부분도 파악 가능한 detecting

![](https://i.imgur.com/296WvML.png)

2d가 잘 안되는 곳은 3d도 실패하더라.. Segmentation이 잘 안되니 Mesh도 잘 안생긴다.

![](https://i.imgur.com/LsZIkxv.png)
