# Lecture 16: Detection and Segmentation

---

# Slow R-CNN Training

---

- 지난 강의에서 했던 Detection을 Training Time 기준으로 더 배운다.

- region proposal method는 selective search라고 있는데 이것 자체는 블랙박스로 둔다. 어차피 후에 NN 방법으로 바뀌었다.

- **참고로 지난 시간에 ground truth box와 비교해서 각 region proposal이 positive인지, negative인지, neutral인지 구분했다.**

- **즉 다음 그림과 같이 object가 있으면 purple, 없으면 red, 있는데 ground truth가 잘 반영이 안됐으면 sky blue로 표현한다.**

- **0.5를 넘으면 positive, 0.3 이하면 negative, 그 사이면 neutral로 두곤 한다.**

![Image](https://i.imgur.com/ParP3Qq.png)

- 이제 neutral은 무시하고, positive함과 negative함을 train한다.

- neutral은 학습시키면 오히려 헷갈린다.

- 일단 Region을 떼어내서 224 \* 224로 만든다.

![Image](https://i.imgur.com/BDRohRK.png)

- 해당 epoch에서 모든 region에 대해서 CNN은 동일한 weight를 가진다.

- 그리고 두가지를 얻는다.
  
      1. class
      2. transform(region -> bounding box)

- positive하면서 GT box(Ground Truth box)와 잘 맞는 RoI의 경우, category label이 GT box와 같을 것이다.

- negative하면서 GT box와 잘 맞지 않는 RoI의 경우, background로 분류해야할 것이다.
+ 앞선 시간에 보았던 region proposal과 GT box의 matching을 통한 IoU 계산과정(mAP)에서 Positive가 나온 것에 GT box의 label을 가져다 붙인다.

---

- 즉, Region과 output bounding box와 class label을 묶는다. output bounding box가 어떤 class가 되어야하는지 정하는 것이다. 이 과정은 training에 들어가기 직전에 해야하는 부분이다.

- 정리하면, region proposal을 offline으로 전체 training set에 대해서 행하고, matching up 과정도 offline으로 전체 training set에 대해서 돌린다.

- 반대로 faster R-CNN의 경우 조금 tricky한 것이, 이 과정에 있어서 online으로 처리가 가능하다. 이는 region proposal을 online으로 training이 가능하기 때문.

---

- 참고로 negative로 나온 region은 background이므로, 어떤 regression target을 가지고 있다는 것이 말이되지 않을 것이다. 그래서 여기선 regression loss가 없다.

- regression loss는 positive region에 대해서만 존재한다.

![Image](https://i.imgur.com/k5tDL6i.png)

# Fast R-CNN Training

---

- 여기서는 이미지를 일단 통째로 Backbone에 넣고 결과물로 나온 feature map에다가 region proposal을 한다.

- 나머지는 기존 R-CNN과 동일

![Image](https://i.imgur.com/GOKASZi.png)

# Faster R-CNN Training : Learnable Region Proposals

---

- 2 stage method이다.

- stage 1 : Anchor -> Region Proposal

- stage 2 : Region Proposal -> Object Box

- anchor : 여러 개의 box가 있는 중심점으로, input image 곳곳에 있다.

![Image](https://i.imgur.com/TQrJ1ut.png)

- 다만 여기서는 positive인지 negative인지 RPN이 판단하게 학습시킨다.

- RPN은 Object인지 background인지 각 anchor마다 예측하고, Region Proposal에 적합한 transform을 예측한다.

- positive, negative, neutral을 구분하는 메커니즘은 앞선 RCNN과 똑같다.

- RPN은 GT box와 matching해서 얻은 positive, negative, neutral 세 가지 anchor를 수도 없이 output한다.

![Image](https://i.imgur.com/cRBgmdD.png)

- 이번엔 RoI와 GT box를 match up 한다.

- 이제 RoI를 crop하고 조절해서 classification과 transform(box target)을 구한다.

- proposal이 RPN으로 되는 것 제외하면 다른 부분은 Fast R-CNN과 똑같다.

- 이 과정은 모두 online으로 해야한다. 왜냐하면 region proposal이 training 동안에 계속 바뀌기 때문이다.

![Image](https://i.imgur.com/O8A4Bjp.png)

# Recap: Fast R-CNN Feature Cropping

---

- Fast R-CNN의 목표 중 하나:
  
  - Region proposal을 crop, resize하는 것을 differentiable한 방법으로 하는 것.

- snap하여서 feature map에 나타난 grid cell의 크기가 일정하지는 않다는 문제가 있다.

- 각 sub region마다 2x2 Max pooling을 갈긴다.

- 그러면 input size의 크기가 다르더라도 Region Feature는 항상 같은 size로 그 결과물이 나온다.

- 그래서 differentiable하기도 하고 그 다음 Network에 input으로 쓸 수도 있다.

![Image](https://i.imgur.com/nT78fWk.png)

- 하지만 두 가지 문제가 있다.
1. snapping 때문에 misaligning이 생긴다.
   
   - snapping이 크게 두가지 방법이 있는데, 하나는 전체의 region proposal을 snap 하는 것이다. 다른 방법은 RoI를 나눠서 snap하는 것이다.
   
   - 위 그림을 보면 blue box 형태로 snap했다가, 분할을 한 후, 그 분할한 상태로 원래 이미지로 다시 돌려보낸다.
   
   - 그리고 input image로 돌아간 각 sub region의 중간점은 midpoint라고 부른다.
   
   - **이 snapping 때문에 input bounding box의 midpoint와는 위치가 다르다.** 이는 RoI Pool operation에서 잠재적 문제가 된다.

2. box coordinate는 backpropagate 할 수가 없다.
   
   - feature map과, crop할 box의 coordinate를 값으로 받는다.
   
   - 하지만 snapping 때문에 coordinate로는 backpropagate할 수 없다. 
   
   - *snap에 해당하는 미분이 있을리가 없지 않는가?* 즉, 반쪽짜리 gradient만 구할 수 있는 것이다.

# Cropping Features: RoI Align

---

- 이 친구가 앞서 본 문제들을 해결한 것.

- operation에서 snapping을 없애버렸다.

- project하는 것까지는 똑같으나 이미지의 크기를 grid에 맞추는 그런거는 없다. 즉, snapping은 없다.

- subregion의 크기를 통일 시킨다. 그 결과 grid 위에 중간점이 없을 수도 있다.

- 그리고 subregion 사이에 간격이 같은 sample을 여러개 만든다.

- 이 projection 결과가 grid 위에 있지 않으므로, 각 sample은 real value로 bilinear interpolation으로 구해야한다.

![Image](https://i.imgur.com/H9QuSJF.png)

- 계산 순서는 저번 강좌 때처럼 진행된다.

- 여기서 x, y는 초록색 점의 위치이고, i, j는 그 주변의 gridpoint의 것이다.

![Image](https://i.imgur.com/JDsbvMZ.png)

![Image](https://i.imgur.com/yd4PC7z.png)

![Image](https://i.imgur.com/5xaTtsZ.png)

![Image](https://i.imgur.com/ffYzEJ4.png)

- 이렇게 해서 sample 4개를 구하고 이를 MaxPooling 한다.

- 이 결과는 differentiable하다. upstream gradient가 제일 하단의 gridpoint 변수까지 잘 도달할 것이다.

- 그래서 이제 feature vector in the grid에도, bounding box의 actual position에도 그 gradient를 전달할 수 있다.

![Image](https://i.imgur.com/pMgXjXq.png)

- output feature가 input box에 정렬되었다! 이제 coordinate를 backpropagtion으로 구할 수 있다!

![Image](https://i.imgur.com/sF4Txkh.png)

# No-Anchors Detection

---

- Anchor Box를 안쓰는 방법이 있을까?

![Image](https://i.imgur.com/01eRo9D.png)

## Detection without Anchors: CornerNet

- 이걸 해낸 연구가 있다.

- bounding box의 parameter를 좌측 상단 좌표, 우측 하단 좌표로 둔다.

- 그리고 각 픽셀별로 해당 object category에 대해서 좌측 상단 좌표/ 우측 하단 좌표일 확률을 기록한다.

- 그리고 이를 heatmap으로 각각 작성한다.

- training은 각 pixel에 대해서 crossentropy loss로 행하면 된다.

![Image](https://i.imgur.com/7zi74nc.png)

- 그럼 이제 각각의 Corner를 어떻게 매칭시켜서 박스를 만드나?

- 이는 또 따로 각 corner에 대해서 embedding vector를 만들어서 해결한다.

- 즉, 우측하단 임베딩 벡터와 좌측상단 임베딩 벡터가 서로 비슷해야 하나의 박스를 형성한다는 것이다.

# Semantic Segmentation

---

- 모든 pixel을 category label을 붙이겠다는 뜻

- 객체는 상관없이, pixel에만 집중하는 것이다!

![Image](https://i.imgur.com/y01086d.png)

## Semantic Segmentation Idea: Sliding Window

- 연산량이 너무 많다. 그리고 겹치는 feature를 재사용하지 못한다.

![Image](https://i.imgur.com/M0uwvis.png)

## Semantic Segmentation: Fully Convolutional Network

- **3 \* 3, stride = 1, padding = 1인 것을 stack한다면, input과 output의 spatial size는 같다.**

- **그리고 output의 channel을 class의 개수와 동일하게 만들 것이다.**

- 그렇게 해서 나온 output의 하나의 픽셀에서 벡터하나를 꺼내면 channel vector가 나올 것이다. 그리고 이 channel vector의 각 channel을 score로 보겠다는 것이다.

- 그리고 각각의 pixel에 대하여 Cross Entropy Loss를 구할 수 있다.

- detection과는 다르게, input size에 따라서 output size가 고정적이다.

![Image](https://i.imgur.com/zHSTsht.png)

### Question : 어떻게 클래스의 개수를 파악하나요?

- 미리 선정해놓고 해야한다. 이걸 자동으로 할 수는 없다.

- VGGNet에서 봤듯, 3\*3 Convnet을 계속 쌓으면 receptive field가 1+2L씩 커진다. 즉, 큰 이미지를 요구하게 된다.

- Segmentation은 기본적으로 resolution이 높다. 근데 convolution을 high resolution에 하면 비용이 많이 든다. (그래서 ResNet에서 aggressive downsample을 하는 것.)

![Image](https://i.imgur.com/XUZW4zK.png)

- 그래서 downsampling과 upsampling을 한다.

![Image](https://i.imgur.com/nFijB7u.png)

- 근데 Upsampling이 뭔데?

# In-Network Upsampling: “Unpooling”

- 아래는 안쓰는 옛날 방법론들

![Image](https://i.imgur.com/zuUY5zN.png)

# In-Network Upsampling: Bilinear Interpolation

- 앞서 봤던 Bilinear Interpolation 방법을 쓸 수도 있다.

- 가까운 **두 곳의 neighbor**를 이용해서 **linear approximation**을 한다.

![Image](https://i.imgur.com/PQKZ6e1.png)

# In-Network Upsampling: Bicubic Interpolation

- 이미지 프로세싱하면 잘 나오시는 분. 가장 자주 쓰인다고 한다.

- 가까운 **세 곳의 neighbor를 이용**해서 **cubic approximation**을 한다.

![Image](https://i.imgur.com/p3aOkKs.png)

# In-Network Upsampling: “Max Unpooling”

- Max pooling의 반대.

- 여기서부터는 Net에 의존성이 있는 Layer가 된다. 즉, downsampling할 때 사용했던 부분과 연관성(데이터를 기억해서 다시 사용한다는 의미이다. sampling 위치 등)을 띄게 된다.

![Image](https://i.imgur.com/FrcV2Vt.png)

# Tips

- 어떤 upsampling algorithm을 고를까 <- 어떤 downsampling algorithm을 선택했냐

- Average Pooling 을 했었다면, Nearest Neighbor, Bilinear, Bicubic

- Max Pooling을 했었다면, Max Unpooling을 해라.

# Learnable Upsampling: Transposed Convolution

- 앞서 배운 upsampling은 learnable parameter가 없었다.

- 이것은 있다! 그래서 학습을 할 수 있다는 것. 방법을 보자.

- Convolution은 기본적으로 stride를 늘림으로써 downsampling을 할 수 있음을 알 수 있다.

![Image](https://i.imgur.com/F4pX2yO.png)

- 이제는 stride = 2로 늘렸다.

![Image](https://i.imgur.com/tukh2Og.png)

- 즉, stride > 1이면 Learnable Downsampling이 된다는 것.

- 그럼 stride < 1이면 Learnable upsampling이 될 수도 있다는 상상을 할 수도 있겠다.

- 3\*3 filter에 input tensor의 scalar 값을 곱한다. 그리고 이를 output에 다음 그림과 같이 복사한다.

![Image](https://i.imgur.com/N8wAVLF.png)

- 아래와 같이 겹치는 부분은 두 output을 더하면 된다.

![Image](https://i.imgur.com/M0BBZw5.png)

- 원하는 크기의 output이 되도록 마지막에는 잘라주도록 한다.

![Image](https://i.imgur.com/1MayF0r.png)

# Transposed Convolution: 1D example

- 1D 상황에서의 Transposed Convolution을 살펴본다. input에 filter를 곱한 것을 output에 붙여넣는다고 생각하면 되겠다.

![Image](https://i.imgur.com/EWB0BoN.png)

- 이름이 좀 많다.
  
  - Deconvolution
  
  - Upconvolution
  
  - Fractionally strided convolution
  
  - Backward strided convolution
  
  - Transposed Convolution

# Convolution as Matrix Multiplication (1D Example)

왜 Convolution이라는 이름이 붙었을까

- Convolution을 행렬곱으로 표현하면 다음과 같다.

- $a$는 filter다.

![Image](https://i.imgur.com/Gu0Ybh6.png)

- 이것이 Transposed convolution이다. 기존 것과는 다르다는 것을 확인할 수 있다.

![Image](https://i.imgur.com/uNvyamc.png)

- 다음은 stride가 1 이상일 때의 상황이다.

- 방금 보았던 stride 1일 때와는 다르게 0이 아닌 원소들의 sparsity pattern이 stride에 따라서 크게 달라진다는 것을 알 수 있다.

![Image](https://i.imgur.com/sIgMYDJ.png)

- 즉, Transposed 된 것의 convolution은 normal convolution으로 표현할 수 없다는 것을 알 수 있다.

- Transpose convolution의 forward pass가 Normal Convolution의 backward pass이다.

# 다시 돌아온 Semantic Segmentation: Fully Convolutional Network

![Image](https://i.imgur.com/FGvtR5G.png)

- 아직 몇 가지 문제가 남아있다.

- 픽셀별로 class는 구분할 수 있지만 그걸로 object를 detect할 수는 없다는 것.

- 픽셀 단위로 object를 detect하는 것을 하고 싶다.

![Image](https://i.imgur.com/aCemBlO.png)

- 보통 컴퓨터 비전에서

- Things = object instance로 치는 카테고리 = 개, 고양이, 캔 등...

- Stuff = object instance로 안치는 카테고리 = 하늘, 풀, 물, 나무 등...

- 다음 그림에서 Object Detection은 Things만 찾고, Semantic Segmentaion은 Things와 Stuff를 동시에 찾는다는 것을 알 수 있다.

![Image](https://i.imgur.com/9sVScgB.png)

# Computer Vision Tasks: Instance Segmentation

- Object Detecting 이후에, 각 pixel에 segmentation masking을 하는 작업을 의미한다.

![Image](https://i.imgur.com/Zg29f9R.png)

- 이걸 해낸 알고리즘이 있었으니.

# Instance Segmentation: Mask R-CNN

- Mask Prediction이 추가된다!

- 이 branch는 object의 background와 foreground를 구분하는 기능을 한다.

![Image](https://i.imgur.com/vy6U7vu.png)

- 전체 흐름은 전과 똑같다. 다만 마지막에 segmentation을 masking하는 단계가 있을 뿐이다.

![Image](https://i.imgur.com/EdhOMaO.png)

- 이제 object detection과 instance segmentation을 동시에 할 수 있게 되었다!

- 이렇게 좋은 결과가 나온다!
  
  - 다른 RoI를 주면 결과가 달라진다
  
  - 다른 Category를 주면 결과가 달라진다

![Image](https://i.imgur.com/LXi17IO.png)

![Image](https://i.imgur.com/ZniuV0q.png)

# Beyond Instance Segmentation: Panoptic Segmentation

- Things에 대해서만 번호를 붙여 구분하는 Segmentation이다.

![Image](https://i.imgur.com/F9L1g5h.png)

# Beyond Instance Segmentation: Human Keypoints

# Mask R-CNN: Keypoint Estimation

----

- 사람의 행동을 컴퓨터로 따라하고 싶다.

- 사람 몸의 곳곳을 keypoint로 사용한다. 그를 통해서 취하고 있는 자체를 예측한다.

![Image](https://i.imgur.com/XCKzOug.png)

- Segmentation mask 대신에 Keypoint Mask를 predict한다.

![Image](https://i.imgur.com/grO0wW5.png)

- Object Detection, Keypoint Estimation, Segmentation을 합하면 다음과 같이 사용할 수 있다.

![Image](https://i.imgur.com/Slf9yNA.png)

# General Idea: Add Per-Region “Heads” to Faster / Mask R-CNN!

- 지금까지의 경험으로 보아, 아래 표시된 Head 부분만 교체하면 무엇이든 할 수 있을 것만 같다.

![Image](https://i.imgur.com/y40Wv7M.png)

# Dense Captioning

- 그래서 위를 때고 LSTM을 넣어서 Image Captioning을 한다. region마다 caption을 예측하는 것.

![Image](https://i.imgur.com/0zMwcp1.png)

- 그 결과

![Image](https://i.imgur.com/yBgjPgP.png)

# 3D Shape Prediction

- 이번에는 region마다 3D triangle mesh를 예측하겠다는 것.

- Mask R-CNN + Mesh Head

![Image](https://i.imgur.com/u3l8mcy.png)

- 그 결과

![Image](https://i.imgur.com/8phWsTM.png)

# Own Question

anchor box -> Region Proposal -> Negative/Positive 판단

---

## Online vs Offline learning

### Offline learning

배치 학습에서는 시스템이 점진적으로 학습할 수 없습니다. 가용한 데이터를 모두 사용해 훈련시켜야 합니다.

이러한 방식은 시간과 자원을 많이 소모하여 일반적으로 오프라인에서 가동됩니다.

먼저 시스템을 훈련시키고 제품 시스템에 적용하면 더 이상의 학습 없이 실행됩니다.

즉, 학습한 것을 적용할 뿐입니다. 이를 오프라인 학습(Offline Learning)이라고 합니다.

### Online learning

온라인 학습(Online Learning)은 그 반대입니다. 점진적으로 학습하는 것이지요.

온라인 학습에서는 데이터를 순차적으로 한 개씩 또는 미니배치(Mini-Batch)라 부르는 작은 묶음 단위로 주입하여 시스템을 훈련시킵니다.

매 학습 단계가 빠르고 비용이 적게 들어 시스템은 데이터가 도착하는 대로 즉시 학습할 수 있습니다.

온라인 학습은 연속적으로 데이터를 받고 빠른 변화에 스스로 적응해야 하는 시스템에 적합합니다.

컴퓨팅 자원이 제한된 경우에도 적합하다고 할 수 있습니다. 학습이 끝난 데이터는 더 이상 필요하지 않으므로 버리면 됩니다.

## Anchor Box

**Anchor Box는 input image 전체에 놓는가?**

https://d2l.ai/chapter_computer-vision/anchor.html

모든 pixel에 anchor를 놓는다.

결국 하나의 픽셀에는 $n+m-1$개의 box가 생긴다. 한 이미지에 생성되는 anchor box는 총 $wh(n+m-1)$개인 것이다.
