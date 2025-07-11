# 문수영 교수님의 드론 sw + Sciklit-learn 강의
강의 자료는 교수님 깃허브 링크에서 다운가능.<br>
https://github.com/sooyoungmoon/scikit_learn

## 이론강의 (깃허브에 정리후 메일로 제출)
1. 머신러닝 (ML) 
- 정의 : 인공지능의 한 분야로, 사람이 프로그래밍 하지 않고 컴퓨터가 데이터 입력으로 스스로 학습하는 기술.
- 학습과 추론과정
  - 준비된 학습 데이터 -> 머신러닝 -> 모델 완성 (모델학습)
  - 입력 데이터 -> 모델 -> 결과 (추론)
- 머신러닝 과정 : 데이터얻기 = 라벨링 -> 전처리, 문자-숫자변환, 중복제거등 -> 모델학습 -> 데이터를 모델에 입력해 테스트, 충분히 학습됬는지 판단 -> 성능 피드백
- 머신러닝 알고리즘에 따른 모델 분류
  1. 지도학습 : 라벨링된 데이터를 학습하여 정답에 맞게 도출
    ex) 스팸메일 분류문제, Support Vector Classifier SVM : 클래스 구분선과 인접 샘플들 간 거리가 최대가 되도록 분류하는것.
  2. 비지도학습 : 라벨이 없는 데이터를 학습하여 패턴이나 구조를 발견
  3. 강화학습 : 라벨이 없는 데이터를 학습하여 출력에 대한 평가를 피드백으로 사용해 개선해나감
- 머신러닝 모델 ≠ 머신러닝 알고리즘
  > 알고리즘은 모델을 만들어내는 기법이지, 모델자체가 아니다.
- 언더피팅과 오버피팅 : 모델이 지나치게 단순하여 패턴을 파악하지 못하거나, 반대로 지나치게 복잡하여 새로운 데이터에 대해 낮은 성능을 보이는 현상
- 머신러닝의 장점 : 대규모 + 다차원 데이터 처리에 유리하고, 데이터가 쌓일수록 성능이 향상되며, 숨겨진 패턴을 찾아내기에 좋다.

2. 사이킷 런 (Scikit-learn)
  https://github.com/sooyoungmoon/scikit_learn/blob/main/notebooks/scikit_learn_quickstart.ipynb
- 파이썬 기반의 머신러닝 오픈소스 라이브러리
- 분류, 회귀, 클러스터링 등의 알고리즘을 간편하게 구현가능
- numpy, Scipy, pandas 등의 파이썬 라이브러리와 연동되어 편리
  
3. Estimater : fitting과 라벨링을 맡음. 여러 종류가 있으며, clf.fit(x,y) 등의 clf. 명령어로 사용
  (Scikit-learn) 홈페이지에서 자세한 설명이 나와있다.
- Transformer와 pre-processors : 숫자를 문자로 변환하는 .transfer(), 중복제거 등의 전처리와, 숫자 데이터의 경우 평균이 0이고 표준편차가 1을 갖도록 데이터 정규화 담당.
  (pre-processors는 Transformer의 일종)

4. Data-pipeline
- 데이터를 입력하면, 빠진 데이터를 추가하거나 정규분포로 변환하거나 등의 전처리 단계를 거쳐 추론까지 진행하는 Data flow를 지나, 최종 결과에 도착한다.
  <img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/3608d82e-cd8f-4f73-9e1c-98e643af2e81" />

5. Cross validation
- 내가 고른 데이터가 유독 좋아서 결과가 잘 나온걸수도 있다.
- 이를 방지하기위해 (Q. 모델이 처음 보는 데이터를 입력 받았을 때 얼마나 일관된 추론 성능을 보이는가?)를 검사함
  <img width="537" height="771" alt="image" src="https://github.com/user-attachments/assets/2b186c11-f2b6-44c3-b733-c506786d37c9" />
- 데이터 전체를 k개 묶음으로 나누고, k번째 묶음만을 테스트에 사용할때 나머지는 모델학습에 사용하면 된다.

## 지도학습 이어서
- KNN (K Nearest Neighbors) : 지도학습과 비지도학습 모두 사용가능한 기법이고, 거리위주 가중치를 선택하면 노란 영역에 있어도 초록색과 가장 근접하니 초록색으로 분류될수도 있다는 뜻.
- 예측 정확도와 민감도 (Precision and Recall) : 성능지표로 쓰임.<br>예측 정확도 = 가짜로 예측한게 진짜일 확률을 "열에서만" 계산함.<br>민감도 = 가짜를 검출해내지 못한 확률을 "행에서만" 계산함.<br>
  <img width="1527" height="795" alt="image" src="https://github.com/user-attachments/assets/b97f145b-e7cf-42c8-a63d-7c79dd1d16ba" />
- 둘다 높으면 베스트지만 힘들기에, F1-Score를 보며 밸런스를 조절해야한다.
  
  <img width="1357" height="468" alt="image" src="https://github.com/user-attachments/assets/11b0009a-00db-48b1-a4fd-113668f090d6" />
  
- 예제1. 손글씨로 쓴 숫자 식별 문제<br>
  <img width="795" height="210" alt="image" src="https://github.com/user-attachments/assets/2703b8b5-0398-474d-bac7-e2ea7288d900" />
- https://github.com/sooyoungmoon/scikit_learn/blob/main/notebooks/classification_hand_written_digits.ipynb
