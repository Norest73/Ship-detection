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
