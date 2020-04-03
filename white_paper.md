1) Introduction to Solution
     - Abstract (전체 요약)

2) Data Processing Techniques
     - 해당 대회의 데이터셋은 전부 (3000, 3000)의 동일한 사이즈의 인공위성 이미지이며, 이는 GPU를 통한 이미지 처리를 진행하기에 굉장히 큰 크기에 속합니다.

     - Image Split/Merge (3000 → 1024 → Model → 3000)
     - Image Split시 기존 대비 0.7이상 넓이가 줄어든 경우 제외
     - Class balancing (대략 1 : 0.9 : 0.8 : 0.4)
     - Data Augmentation (Random, 각도 90도/180도/270도)

3) Details on Modeling Tools and Techniques
     - Gliding Vertex

4) Conclusion and Acknowledgments
     - 간단한 방법으로 각도의 위험성 없애고, 기존 모델의 성능을 끌어올린 민타우 최고!

5) Appendix (pre-trained model 학습에 사용된 public 데이터셋 리스트 및 추가 보충 자료)
     - Detectron Faster-RCNN (ImageNet으로 학습된 Resnet101)
          https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md


Introduction to Solution
객체 인식 알고리즘은 원본 이미지와 수평한 직사각형 기준으로 영역을 산출하는 HBOX(Horizontal Bounding Box) 방식을 기반으로 발전해왔습니다. 딥러닝이 적용된 이후 객체 인식 알고리즘은 급격한 발전을 이루었지만 이는 마찬가지입니다. 그러나 HBOX 방식의 객체 인식은 종종 적합하지 않은 경우가 있습니다. 특히 객체가 작고 밀집되어 있는 경우에는 이것이 객체를 정확하게 인식하는데 큰 문제가 되기도 합니다. 
이 문제를 해결하기위해 HBOX를 여러 각도로 회전시키는 RBOX(Rotated Bounding Box) 방식이 대두되었습니다. RBOX 방식은 HBOX를 회전시켜 객체를 보다 세밀하게 인식할 수 있습니다. 비록 각도가 조금만 차이나도 영역이 크게 어긋난다는 것은 큰 단점이지만, 그래도 RBOX 방식의 알고리즘은 여러 경우에 HBOX보다 효과적입니다.
 본 대회는 위성 이미지를 대상으로 하는 경우로 위 경우에 해당합니다. 데이터셋에 있는 선박들은 다양한 크기를 가지고 있고, 이중에는 작은 배가 밀집되어 있는 경우도 많이 있어 HBOX보다 RBOX가 효과적입니다. 
 본 백서에서는 RBOX를 산출하는 새로운 방법을 통해 작은 배들을 효과적으로 인식하는 알고리즘을 제시합니다. 제시하는 방법의 가장 큰 특징은 각도를 전혀 사용하지 않는다는 것이며, 이로 인해 각도 차이로 RBOX 영역이 크게 벗어나는 위험성을 개선할 수 있었습니다. 또한 HBOX와 연계하여 RBOX를 산출함으로써 보다 안정적으로 객체 영역을 인식합니다.

Data Processing Techniques
 본 대회의 데이터셋 위성 이미지들은 2가지 특징을 가지고 있습니다. 하나는 이미지 사이즈가 전부 (3000, 3000)로 굉장히 크다는 것이며, 다른 하나는 데이터셋의 클래스들이 굉장히 불균형 하다는 것입니다. 이 문제점들을 해결하기 위해 본 백서에서는 총 3가지의 이미지 처리방법을 사용합니다.
 첫 번째는 이미지 분할 및 병합입니다. (3000, 3000) 사이즈의 원본 이미지들을 (1024, 1024) 사이즈로 분할하여 모델 학습을 진행합니다. 이때 중요한 것은 분할 이미지들끼리 어느 정도 겹치도록 하여 객체 손실을 최소화 하는 것입니다. 이를 위해 분할된 학습 이미지들이 200씩 오버랩되도록 조정하였습니다. 또한 분할하면서 원본 대비 객체가 70%보다 손상된 경우에는 이를 데이터에서 제외 하였습니다. 테스트 이미지들은 객체 손상을 최대한 막기 위해 500씩 오버랩되도록 분할하였으며, 분할된 이미지들에 대해 각각 객체를 예측하고 R-NMS를 통한 병합을 진행 하였습니다.
 두 번째는 오버샘플링을 통한 클래스 밸런싱입니다. 학습 데이터에서 4개 클래스들의 객체 비율은 각각 67.3%, 22.4%, 10.1%, 0.2%로 굉장히 불균형적이며 이는 올바른 모델 학습에 방해가 됩니다. 클래스 균형을 맞춰주기 위해 'container', 'oil tanker', 'aircraft carrier' 클래스객체를 각각 3배, 6배, 100배로 오버 샘플링하여 클래스 밸런싱을 진행하였습니다.
 마지막은 이미지 어그멘테이션입니다. 위성 이미지 특성상 다양한 어그멘테이션을 사용할 필요는 없지만, 특정 각도에 학습이 편중되지 않도록 50% 확률로 랜덤하게 90°, 180°, 270°로 회전 어그멘테이션을 적용하였습니다.

Details on Modeling Tools and Techniques
 기존 RBOX (Rotated Bounding Box) 기반 객체 검출 모델은 각도의 정확도에 따라 성능이 좌지우지 되는 경우가 많았습니다. 특히 선박과 같이 긴 객체를 인식할 때 이 문제점은 더욱 두드러져 모델의 성능을 떨어뜨리는 주 원인중 하나였습니다.
 저희 팀은 이 문제점을 해결하기 위해 Gliding Vertex on the horizontal bounding box 기법을 활용하였습니다. Gliding Vertex 의 가장 큰 특징은 각도를 전혀 사용하지 않고 HBOX(Horizontal Bounding BOX)로부터 RBOX를 추출해낸다는 것입니다. 대신 HBOX 꼭지점으로부터 RBOX 꼭지점까지의 거리 비율(α1, α2, α3, α4)을 산출합니다. <그림 1>은 α를 통해 HBOX로부터 RBOX를 산출하는 방식을 잘 보여줍니다. 
 
<그림1>
 또한 모든 경우에 RBOX를 사용하는 것보다, 기준을 두고 선별적으로 HBOX도 사용하는 것이 모델의 성능을 향상시킬수 있습니다. HBOX의 크기 대비 RBOX의 크기 비율을 나타내는 Obliquity factor(r)를 산출하고, 기준(Threshold)을 넘어서는 경우에는 RBOX 대신 HBOX를 결과물로 선정하였습니다. <그림2>는 HBOX와 RBOX의 Obliquity factor를 산출하고 최종 결과물을 선정하는 과정을 보여줍니다. 
 
<그림2>
 
.<그림3>
 Detector 모델은 가장 대중적으로 활용되는 ResNet-101 Backbone 기반 Faster-RCNN을 사용하였습니다. 다만 차이점은 위에서 HBOX기반 모델과 달리 (x, y, w, h, α1, α2, α3, α4, r) 총 9개의 변수를 결과물로 산출한다는 점입니다.

Conclusion and Acknowledgments
 이 객체 검출 방법의 가장 큰 특징은 Detector 모델로 대중적으로 활용되는 Faster-RCNN을 사용했음에도 불구하고, RBOX를 색다르게 산출하는 방법으로 성능을 확연하게 끌어올렸다는 점입니다. Gliding Vertex 를 통해 각도를 전혀 사용하지 않고 RBOX를 산출하는 방식은 보시다시피 선박과 같이 긴 객체를 대상으로 할 때 더욱 효과적이었습니다.  Gliding Vertex 의 다른 장점은 바로 다른 Detector 모델에도 적용할 수 있다는 것입니다. R2CNN이나 RoI-Transformer과 같은 모델에도 적용할 수 있으며, 심지어 RetinaNet, EfficientDet과 같은 one-stage 모델에도 적용이 가능해 적절하게 활용한다면 성능을 크게 향상시킬 수 있을 것입니다.
 본 대회를 마무리하면서 가장 아쉬웠던 점은 시간 부족이었습니다. 방법론에 대한 리서치가 늦어진 만큼 다양한 모델에 테스트해볼 수 있는 시간이 부족했습니다. 또한 선박 크기가 천차만별하게 제공되는 대회에서 객체 크기에 따라 다르게 구분할 수 있는 모델들을 학습시켜 앙상블 시키는 아이디어를 적용해보지 못했다는 점이 아쉽습니다. 다만 Gliding Vertex 를 실제로 적용하여 효과성을 확인할 수 있었다는 점은 만족스럽습니다. 마지막으로 대회 준비 및 참여자 지원에 힘써주신 데이콘 직원분들에게 진심으로 감사의 말씀 올립니다.

Appendix. Pretrained Models
 Faster-RCNN의 성능향상을 위해 ImageNet 데이터셋을 통해 사전학습된 ResNet-101 모델을 사용하였습니다. (URL: https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-101.pkl)











