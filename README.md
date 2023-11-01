## 🏆 수상
2023 한성대학교 캡스톤디자인 | 장려상

<br><br>
## :raised_hands: 소개
ResNet-34 아키텍처를 기반으로 하여, 스펙트로그램의 시각적인 특징을 추출하는 **손글씨 음향 신호 감지를 통한 문자 인식 모델**입니다.


<br><br>
## 💪 주요 기능
1. PyTorch에서 GPU 사용 환경을 설정합니다.
2. 입력 이미지에 대한 데이터 전처리 및 정규화를 정의하는 변환을 구성합니다.
3. 데이터셋을 정의하고, 데이터 변환을 적용합니다.
4. ResNet-34 아키텍처를 사용하여 이미지 분류 모델을 생성합니다.
5. L2 정규화를 적용합니다.
6. K-Fold 교차 검증을 통해 모델을 학습하고 성능을 평가합니다.
7. 폴드별 최고 정확도를 가진 경우 모델 가중치를 저장합니다.
8. 학습 및 테스트 결과를 파일에 저장합니다.
   

<br><br>
## 🦾 주요 기술
**Model - CNN**
* PyCharm IDE
* Python 3.9.13
* Scikit_learn 1.3.1
* Torch 1.13.1
* Torchvision 0.14.1

<br><br>
## 🧬 모델 아키텍처
<div align="center">
  <img width="60%" alt="image" src="https://github.com/CAP-JJANG/.github/assets/92065911/7fcd5810-2541-4a52-a0aa-a758c61e8fc8">
</div>


---
This repository serves as the foundation of [CSD-Model](https://github.com/CAP-JJANG/CSD-Model)
