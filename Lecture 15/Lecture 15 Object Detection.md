# Lecture 15 : Object Detection

![Image](https://i.imgur.com/47RlUFe.jpg)

그동안은 Image Classification이었다.

이번에는 Object Detection = 어디에 뭐가 있다!를 할 것이다.

사진 내부에서 어느 위치에 있는지 찾는 task는 크게

1. Object Detection
2. Instance Segmentation

등이 있다.

# Object Dectection의 정의

category label과 bounding box를 output으로 내놓으면 그것이 object detection인 것이다.

다만 이 category가 이미 주어진 fixed set이어야한다.

boundary box는 x, y, w, h 네가지로 구성된다. 기본적으로는 x,y축에 평행한 박스로 한다.

![Image](https://i.imgur.com/U51WMt3.jpg)

아래 그림은 Object Detection의 여러가지 문제들

1. 여러 가지 class를 내놓아야한다.
2. category와 box. 두가지를 내놓아야 한다는 문제
3. 이미지가 전반적으로 커진다는 문제

![Image](https://i.imgur.com/lPNJIOU.jpg)

# Detecting Single Image

하나는 image classification으로 뺀다.

다른 하나는 bounding box로 뺀다.

하나의 scalar로부터 gradient descent가 시작해야하므로, 두 가지 loss를 weighted sum해서 하나의 loss를 구한다. 이런 상황에서 총합으로 나오는 Loss를 **Multitask Loss**라고 부른다.

backbone Net(여기서 CNN을 의미)은 Transfer Learning을 하고 fine tuning을 해서 사용한다.

![Image](https://i.imgur.com/WtY5u3T.jpg)

하지만 object가 하나보다 많은 경우는?

# Detecting Multiple Image

위의 모델을 그대로 사용하는 경우, output이 너무 많이 늘어난다.

![Image](https://i.imgur.com/mmnfbVn.jpg)

## Sliding Window

한 구역씩 옮기면서 Classification을 행하는 방법이 있다.

![Image](https://i.imgur.com/cmaFLQa.jpg)

하지만 그 계산양이 너무나도 많고, 특정 영역에서는 거의 항상 하나의 classification만 거의 무한히 거쳐야한다.

이를 방지하기 위해서 object 전체를 포함할 가능성이 높은 곳을 제안하는 모듈을 사용한다.

# Region Proposal

초창기에는 특수한 방법을 사용했지만, 현대에는 아예 러닝 기반의 알고리즘을 사용한다.

![Image](https://i.imgur.com/UVdGZ2q.jpg)

# R-CNN : Region Based CNN

RoI(Region of Interest)를 뽑고, 이를 고정된 크기로 만든 후, ConvNet에 forward pass한다. 그리고 C + 1개(배경까지 포함해서) class를 얻어낸다.

하지만 이러면 Region을 설정하는 데 있어서 아무런 러닝 요소가 없다. 즉, Region 설정에 발전이 없다는 뜻이다. 

그래서 일단 class 결과와는 별개로 final box(Bounding box라고 금발근육남이 후에 말함)를 따로 뽑아낸다.

여기서 input region을 final box로 만드는 함수를 transform이라고 한다. 이 transform을 regression을 통해서 찾아나간다.

bounding box가 4가지의 수 t로 표현되므로, 이 transform(delta라고 표현)를 4가지 수로 표현 가능하다.

output box $(b_x, b_y, b_h, b_w)$은 CNN output의 transformation과 region proposal의 위치 정보를 합한다. 그림의 수식과 같이. 수식에 따르면 transformation의 output은 ConvNet을 통해서 warp한다는 것과는 무관하다.

$b_x$기준으로 보면, $t_x = 0$이면 input bounding box를 그대로 사용하라는 의미이고, $t_x = 1$이면 width만큼 평행이동하라는 의미이다.

scale은 logarithmic하다.

![Image](https://i.imgur.com/rbwmIR0.jpg)

조금 더 살펴보자.

Test time에서 R-CNN은 다음과 같은 과정을 거친다.

![Image](https://i.imgur.com/VUeIApw.jpg)

3번 과정에서, 파이프라인Threshold 이상이면 출력, 이하면 무시하는 방법Convnet의 Weight는 모두 동일해야한다. 혹은 확률이 높은 K개의 proposal만 남기는 방법이 있다.

참고로 이 방법에서 학습은 좀 어렵다. 한 이미지에 서로 다른 RoI, 다른 이미지의 서로 다른 RoI를 배치로 만들어야하기 때문. 그래서 training time에는 RoI의 batch와 output classification score, transforamtion parameter를 output으로 내놓는다. 그리고 그 둘의 Loss를 합쳐서 하나의 loss scalar로 만들고 gradient한다.

참고로 Threshold, K proposal은 test time에 하고, train 중에는 모든 roi를 학습한다.

# Comparing Boxes: Intersection over Union (IoU)

Roi Loss를 수치화할 필요가 있다.

prediction과 ground-truth box를 어떻게 비교할 수 있나?

그래서 intersection of union. IOU를 정의한다. Jaccard similarity, Jaccard index라고도 불린다.

![Image](https://i.imgur.com/GbsbKUz.jpg)

prediction bounding box와 Ground Truth box를 서로 비교한다.

![Image](https://i.imgur.com/MJxNEPd.jpg)

두 박스 간의 거리를 측정하는 Intersection over Union(IoU)를 정의한다. 수식은 위의 그림과 같다.

IOU 자체는 '오랜지색 넓이 / 보라색 넓이'로 계속한다.

![Image](https://i.imgur.com/DyxNVrC.jpg)

![Image](https://i.imgur.com/OLxQn3C.jpg)

![Image](https://i.imgur.com/qHxlbhP.jpg)

보통 0.5를 넘으면 쓰기에 괜찮다고 생각하는 정도다. 0.7이면 매우 좋고 0.9면 그냥 다 맞다고 봐도 될 정도

# Overlapping Boxes : non-Max Suppression(NMS)

보통 하나의 object에 하나의 box를 내놓지 않는다. 게다가 Object가 많아서 여러 박스들이 필요한 경우, 오버랩이 발생한다.

이러면 IOU를 측정하기 힘든데, 이를 위해 이 오버랩을 없애주어야 한다. (post-processing)

NMS모드에서는 가장 좋은 확률의 박스 하나 선정하고, 다른 박스랑 비교해서 다른 박스들을 지운다.

![Image](https://i.imgur.com/ojBnHo0.jpg)

남은 박스 중, 가장 높은 확률을 가진 박스를 우선 고른다. 이 경우 블루.

그리고 남은 다른 박스와의 IoU를 구한다. 이 값이 threshold를 넘으면 제거한다. 즉, duplicate로 본다는 뜻.

이를 반복한다...

![Image](https://i.imgur.com/TcN72re.jpg)

![Image](https://i.imgur.com/1Fy5WG9.jpg)

## 문제점

object가 많으면 overlap이 매우 많아지고, 많이 overlapping되는 경우, good box도 없앤다.

![Image](https://i.imgur.com/4qBN7XJ.jpg)


# Evaluating Object Detectors: Mean Average Precision (mAP)

![Image](https://i.imgur.com/UcvIFag.jpg)