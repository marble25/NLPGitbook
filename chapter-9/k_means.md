## K-means
---

K-means는 clustering에 사용되는 전통적인 기계학습 알고리즘입니다.   
K-means는 간단하게 말하자면, centroid(중심점)이 더 이상 움직이지 않을 때까지 반복하는 알고리즘입니다.   

scikit-learn에서 K-means를 사용하는 것은 굉장히 쉽습니다.   
이전 장에서 사용하던 코드에 덧붙여 보겠습니다.   

```
from sklearn.cluster import MiniBatchKMeans, KMeans

minibatch = True
if minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1, init_size=1000, batch_size=1000)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)

km.fit(X)
```

scikit-learn은 mini-batch를 사용하는지 안 하는지에 따른 2가지 구현을 제공합니다.   
이 경우에는 minibatch를 사용해서 cluster가 4개로 이루어진 K means를 구해 보았습니다.   
Cluster별로 top words를 뽑아 보겠습니다.   

```
original_space_centroids = svd.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

for i in range(true_k):
    print('Cluster {}:'.format(i))
    for ind in order_centroids[i, :10]:
        print('  {}'.format(terms[ind]))
```

결과는 다음과 같습니다.   

```
Cluster 0:
  graphics
  space
  image
  com
  nasa
  university
  program
  posting
  images
  file
Cluster 1:
  space
  henry
  toronto
  nasa
  access
  com
  digex
  pat
  gov
  alaska
Cluster 2:
  god
  people
  com
  jesus
  don
  say
  think
  believe
  just
  bible
Cluster 3:
  sgi
  livesey
  keith
  solntze
  wpd
  jon
  com
  caltech
  morality
  moral
```

실제로 이 모델을 통해 데이터를 예측하려고 한다면, 간단합니다.   
pre-processing을 끝낸 후, 다음 코드를 실행해주면 됩니다.   

```
km.predict(X_test)
```

지금까지 한 일은 다음과 같습니다.   
dataset을 load하고, 4가지 카테고리를 고르고, 전처리하고, data를 시각화하고, K-means 모델 학습하고, top words를 출력합니다.   
다음 장에서는 다른 종류의 clustering을 알아보겠습니다.   