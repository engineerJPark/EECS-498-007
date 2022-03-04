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

- 하지만 그 계산양이 너무나도 많고, 특정 영역에서는 거의 항상 하나의 classification만 거의 무한히 거쳐야한다.

- 이를 방지하기 위해서 object 전체를 포함할 가능성이 높은 곳을 제안하는 모듈을 사용한다.

# Region Proposal

초창기에는 특수한 방법을 사용했지만, 현대에는 아예 러닝 기반의 알고리즘을 사용한다.

![Image](https://i.imgur.com/UVdGZ2q.jpg)

# R-CNN : Region Based CNN

- RoI(Region of Interest)를 뽑고, 이를 고정된 크기로 만든 후, ConvNet에 forward pass한다. 그리고 C + 1개(배경까지 포함해서) class를 얻어낸다.
  
  - 하지만 이러면 Region을 설정하는 데 있어서 아무런 러닝 요소가 없다. 즉, Region 설정에 발전이 없다는 뜻이다. 

- 그래서 일단 class 결과와는 별개로 final box(Bounding box라고 금발근육남이 후에 말함)를 따로 뽑아낸다.

- 여기서 input region을 final box로 만드는 함수를 transform이라고 한다. 이 transform을 regression을 통해서 찾아나간다.
  
  - bounding box가 4가지의 수 t로 표현되므로, 이 transform(delta라고 표현)를 4가지 수로 표현 가능하다.

- 그림의 수식과 같이, output box $(b_x, b_y, b_h, b_w)$은 CNN output의 transformation과 region proposal의 위치 정보를 합한다. 수식에 따르면 transformation의 output은 ConvNet을 통해서 warp한다는 것과는 무관하다.

- $b_x$기준으로 보면, $t_x = 0$이면 input bounding box를 그대로 사용하라는 의미이고, $t_x = 1$이면 width만큼 평행이동하라는 의미이다. scale은 logarithmic하다.

- Convnet의 Weight는 모두 동일해야한다.

![Image](https://i.imgur.com/rbwmIR0.jpg)

- 조금 더 살펴보자.

- Test time에서 R-CNN은 다음과 같은 과정을 거친다.

![Image](https://i.imgur.com/VUeIApw.jpg)

- 3번 과정에서, 파이프라인이 Threshold 이상이면 출력, 이하면 무시하는 방법
  
  - 혹은 score이 높은 K개의 proposal만 남기는 방법이 있다.

- 참고로 이 방법에서 학습은 좀 어렵다. 한 이미지에 서로 다른 RoI, 다른 이미지의 서로 다른 RoI를 배치로 만들어야하기 때문.  **기억이 안남**
  
  - 그래서 training time에는 RoI의 batch와 output classification score, transforamtion parameter를 output으로 내놓는다.
  
  - 그리고 그 둘의 Loss를 합쳐서 하나의 loss scalar로 만들고 gradient한다.

- **참고로 Threshold, K proposal은 test time에 하고, train 중에는 모든 roi를 학습한다.**

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

NMS모드에서는 가장 좋은 확률 score의 박스 하나 선정하고, 다른 박스랑 비교해서 다른 박스들을 지운다.

![Image](https://i.imgur.com/ojBnHo0.jpg)

남은 박스 중, 가장 높은 확률 score을 가진 박스를 우선 고른다. 이 경우 블루.

그리고 남은 다른 박스와의 IoU를 구한다. 이 값이 threshold를 넘으면 제거한다. 즉, duplicate로 본다는 뜻.

이를 반복한다...

![Image](https://i.imgur.com/TcN72re.jpg)

![Image](https://i.imgur.com/1Fy5WG9.jpg)

## 문제점

object가 많으면 overlap이 매우 많아지고, 많이 overlapping되는 경우, good box도 없앤다.

![Image](https://i.imgur.com/4qBN7XJ.jpg)

# Evaluating Object Detectors: Mean Average Precision (mAP)

- object detecter가 얼마나 잘하는 지 수치화해서 볼 필요가 있다.

- Classification보다 그 metric이 조금 더 복잡하다. Detection에서는 mAP라는 metric을 사용한다.

- AP = 하나의 카테고리에 대해서 얼마나 성능이 좋은가? = Precision과 Recall Curve 사이의 area

- 2-1은 하나의 카테고리에 대한 예상치를 높은 것에서 낮은 것까지 모두 정렬하라는 의미이다.

- 2-2는 예측 박스와 ground truth 박스의 IoU를 구하는 것이다. 보통은 0.5를 기준으로 한다.

- 여기서 이 기준점 0.5를 넣으면 True flag를 지급한다. 즉, 맞는 detection이라는 뜻. 여기서 True를 받으면 True detection, 아니면 False detection이다.

- PR curve(prediction recall curve)에서...
  
      precision = 맞은 것 비율
      recall = 맞은 것을 찾은 ground truth의 비율

- 이것들을 하나하나 계산하게 된다.

- 예시로 다음과 같은 과정을 거치게 된다.

![Image](https://i.imgur.com/0fZEGMv.png)

- 이 결과 나온 AP가 1에 가까울 수록 성능이 좋은 것이다.

![Image](https://i.imgur.com/UcvIFag.jpg)

- 여기서 AP = 1 인 상황이 나오려면 
  
  - 모든 ground truth 박스에 대해서 모두 IoU > 0.5이어야 하며, 
  
  - true positive보다 false positive가 빠르게 정렬되면 안된다. 
  
  - 사실상 false positive이 없어야한다. 그리고 duplicate detection이 없어야한다..

- 각 클래스에 대한 IoU를 구하고 평균을 낸다.

![Image](https://i.imgur.com/4TR2wXe.png)

하지만 이런 방식의 mAP는 박스를 맞는 위치에 두는데 큰 효과가 없다! 왜냐하면 0.5의 threshold만 쓰니깐... 그래서,

이런 과정을 서로 다른 threshod에 대하여 반복하고, 이걸 또 평균을 낸다.

![Image](https://i.imgur.com/ZoHTyCW.png)

# 다시 R-CNN: Region-Based CNN로 돌아와서

- 문제가 하나 있는데, 매우 느리다는 것... 2000개 가량의 이미지를 다뤄야하기 때문.

- 그래서 CNN과 warping을 뒤바꾸는 것이다.

- 이렇게 하면 많은 computation을 서로 다른 region에 대해서 공유하게 된다.

![Image](https://i.imgur.com/GrplE7H.png)

# Fast R-CNN

- 전체 이미지를 CNN(FC Layer 없음. Backbone Network라고도 한다.)에 넣는다. 그리고 feature map을 얻는다.

- Selective Search 등의 Region Proposal method를 가동한다. 이번에는 crop이 아니라, 그 region을 convolutional feature map에 project한다. 그리고 그 만큼의 feature map을 crop한다.

- 그리고 이를 매우 가벼운 CNN에 넣는다. 그리고 output으로 bounding box와 class score를 얻는다.

- 이는 대부분의 연산이 backbone에서 발생해서 매우 빠르다.

![Image](https://i.imgur.com/8Dd6foQ.png)

- 만약 AlexNet으로 했다고 하면 Conv Layer는 Backbone이 되고, 마지막 FC Layer는 end part가 된다.

![Image](https://i.imgur.com/2GZnVdV.png)

- ResNet이라면 마지막 Conv Layer를 end part로, Residual Block을 Backbone으로 사용한다.

![Image](https://i.imgur.com/zdbGPGD.png)

- 그럼 crop and resize는 어떻게 하는 것인가? backpropagation을 하려면 이들도 미분 가능해야하는데...

- 그 방법이 아래에 있다.

![Image](https://i.imgur.com/8G2sQ5o.png)

# Cropping Features: RoI Pool

- 전체 이미지를 우선 CNN에 넣는다.

![Image](https://i.imgur.com/ocdUqfC.png)

- 그리고 RoI를 feature map에 project한다. 이 때, 이 snap이 완전히 겹치지는 않는다. 

- 이 결과를 feature map 격자에 끼우는 것을 snapping이라고 하는 것이다.

![Image](https://i.imgur.com/9fh9J2Y.png)

- 그리고 서브 region으로 또 분할한다. 2\*2 pooling이라고 한다. 최대한 2\*2에 가까운 grid를 구하는 것!

- 그리고 여기다가 max pooling을 한다.

- 즉, RoI는 변하더라도, 그 output은 변하지 않는다는 것이다! = CNN에 넣어서 계산할 수 있다. Backpropagation을 할 수 있다!

![Image](https://i.imgur.com/Z8XqQZ6.png)

- 하지만 이 green and blue region이 서로 조금씩 다른 것을 개선하기 위해 다른 방법이 제안되었다.

# Cropping Features: RoI Align

- snapping을 하지 않고 linear interpolation을 한다.

- 사진만 간략히

![Image](https://i.imgur.com/Gz8ckZ7.png)

![Image](https://i.imgur.com/zDtOx9x.png)

![Image](https://i.imgur.com/gdoMLlr.png)

![Image](https://i.imgur.com/JDsbvMZ.png)

![Image](https://i.imgur.com/yd4PC7z.png)

![Image](https://i.imgur.com/5xaTtsZ.png)

![Image](https://i.imgur.com/ffYzEJ4.png)

![Image](https://i.imgur.com/2T33JZL.png)

![Image](https://i.imgur.com/OuvfUdt.png)

- 결론적으로 fast R-CNN과 R-CNN의 차이는 다음과 같다.

![Image](https://i.imgur.com/MAvZbzr.png)

- 하지만 fast R-CNN은 대부분의 test time이 region proposal에 사용된다.

- 왜냐하면 Selective Search라고하는 heuristic algorithm에 의해 실행되기 때문이다.

- 그래서 CNN으로 Region Proposal을 하게 된 것이다.

![Image](https://i.imgur.com/jJx7sIr.png)

# Faster R-CNN: Learnable Region Proposals

- 2 stage method이다.

- stage 1 : Anchor -> Region Proposal

- stage 2 : Region Proposal -> Object Box

- Region Proposal을 위해서 RPN을 쓰는 것 제외하면 Fast RCNN과 다를 것은 없다.

![Image](https://i.imgur.com/5MpxWq7.png)

# Region Proposal Network (RPN)

- feature map에 anchor box(bounding box인데 고정적인 size이고, sliding하고 다니면서 feature map에 박힌다.)를 생각해보자.

![Image](https://i.imgur.com/Kpb6e0W.png)

- 이 Anchor boxes 모두를 classifying하는 Conv Net에 넣는다.

- 그리고 이 안에 object가 있는가 없는가 binary classification을 한다. softmax나 sigmoid로.

- 즉, positive/negative score를 내놓는다.

![Image](https://i.imgur.com/zZSHaH6.png)

- 아직 부족하므로,

- trick을 써서 box Transform도 내놓도록한다. (위에서 언급한 transform 구하는 대표적인 방법)

- anchor는 green, region proposal box는 yellow다.

- 이런 box transform은 앞서 본 regression loss로 train 가능하다.  

- box transform의 coordinate도 conv net으로 train한다.

![Image](https://i.imgur.com/MOLlXjp.png)

feature map에서 한 위치에 하나의 anchor는 부족해서, K개의 anchor box를 사용한다.

여기서 각 box의 개수, size, ratio 등은 모두 hyperparamter이다.

그리고 각 anchor마다 softmax loss를 구한다.

![Image](https://i.imgur.com/Qhy7g9Z.png)

# 다시 Faster R-CNN: Learnable Region Proposals

- Faster R-CNN을 사용하려면 네 가지의 Loss를 훈련시켜야한다.

- 각 Loss의 의미
1. anchor가 object인가 background인가

2. raw input -> anchor position하는 transform을 계산

3. 어떤 object인지 Classification

4. proposal box -> object box로 transform하는 것을 계산

![Image](https://i.imgur.com/QVk37UL.png)

# two stage method for Faster R-CNN

- 파란색은 entire image를 다루고 region proposal을 하는 network

- 초록색은 하나의 region으로 classification하고 bounding box regression을 한다.

![Image](https://i.imgur.com/iKge2Hs.png)

그런데 정말 두가지의 stage가 필요한 것인가?

# Single-Stage Object Detection

- RPN과 비슷해보인다.

- anchor가 object인지 아닌지를 다루는 stage가 없이 바로 object를 판단한다.

- 이젠 binary classification이 아니라 multinomial classification을 바로 사용할 수 있다.

- box transform도 하나의 anchor마다 내보낸다.

![Image](https://i.imgur.com/gRd2a3K.png)

- category별로 따로 transform을 내놓으면 조금 더 성능이 좋아진다. (category specific regression)

# 각 Object Detection을 비교

Huang et al, “Speed/accuracy trade-offs for modern convolutional object detectors”, CVPR 2017 읽어볼 것

- Two stage가 Single Stage보다 성능이 좋은 이유
  
  - raw image와 feature map을 더 많이 접하기 때문

![Image](https://i.imgur.com/P2PLmul.png)

아래는 State of Technical Art에 대한 것

![Image](https://i.imgur.com/vzq7zo9.png)

# Own Question

### Anchor

https://han.gl/gRTep

Anchor Box : 현대 Object detection에서는 Region Proposal 방식을 사용한다. 이는 객체가 있을 법한 영역을 미리 찾는 방식으로, 기존의 sliding window 방식보다 연산량이 적은 동시에, 효과적으로 객체를 탐지한다. 앵커 박스는 입력 영상에 대해서 객체가 있을 법한 곳에 설정한 박스이며, 특정 영역을 포괄하는 박스에 객체가 있는지 없는지를 네트워크 학습을 통해 판단하게 된다. 입력 영상에 앵커 박스가 생성되는 위치는 샘플링된 feature map으로 인해 결정된다.

# Reference

너무 많은 trick이 있으므로, 아래를 참고해서 구현하는 것이 좋다.

https://competitions.codalab.org/competitions/20794#results

TensorFlow Detection API: 
https://github.com/tensorflow/models/tree/master/research/object_detection
Faster R-CNN, SSD, RFCN, Mask R-CNN

Detectron2    (PyTorch): 
https://github.com/facebookresearch/detectron2
Fast / Faster / Mask R-CNN, RetinaNet