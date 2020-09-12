## 텍스트 Classifying
---

Classification은 지도 학습 알고리즘입니다.   
지도 학습 알고리즘이라는 것은 정답이라는 '라벨'이 있다는 말입니다.   
즉, document가 어떤 클래스 또는 라벨이 속한다는 정보가 존재한다는 말입니다.   
Classification 문제에서는 우리는 document가 어떤 class나 label에 들어있는지를 알고 있고, 이 정보를 학습에 활용합니다.   

이 전에도 강조했던 것처럼, 우리의 text는 clean하고, vectorize 되어 있어야 합니다.   
우리는 '텍스트 Clustering' 장에서 완료했던 상태로부터 시작하겠습니다.   

여기에서 Naive Bayes classifier과 Support Vector Machine classifier를 사용해서 classification 작업을 할 것입니다.   
아래에 작성한 코드는 굉장히 직관적입니다.   

```
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X, labels)

from sklearn.svm import SVC
svm = SVC()
svm.fit(X, labels)
```

간단한 코드를 통해 naive bayes classifier와 svm classifier를 만들었습니다.   
실제로 예측을 하기 위해서는 다음과 같이 사용하면 됩니다.   

```
gnb.predict(X_test)
svm.predict(X_test)
```

결과는 다음과 같이 나올 것입니다.   

```
array([0, 3, 3, ..., 3, 3, 3])
```
