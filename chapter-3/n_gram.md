## N-gram과 전처리
---

### N-gram
텍스트 데이터를 처리하다 보면, context가 굉장히 중요함을 알 수 있습니다.   
vector representation에서는 word의 count만 알기 때문에 context를 잃어버릴 수 있습니다.   
**N-gram**, 그 중에서도 **bi-gram**이 어느 정도는 이 context를 해결해줄 수 있습니다.   

N-gram은 text에서 연속된 n개의 item을 의미합니다.   
여기서 item이란 문자, 음절, 음소 등이 될 수 있습니다.   
bi-gram이란 *n=2*인 경우를 말합니다.   

bi-gram이 text에서 이전 token이 있을 때 그 다음 token이 나올 확률로 계산될 수도 있고, bi-gram이 pair로 나타날 확률로 계산될 수도 있습니다.   
예를 들어, *New York*나 *Machine Learning*이 bi-gram에 의해 생긴 단어입니다.   
다시 말해서, training data에 따라, 우리는 *New* 다음에는 *York*가 올 확률이 높다는 것을 알 수 있고, *New York* 자체를 하나의 객체로 인식하는 편이 좋겠다고 생각할 수 있습니다.   
우리는 bi-gram 모델을 돌리기 전에 필요없는 stop words를 제거해야 합니다.   
Gensim의 경우 bi-gram 모델은 pair가 나타날 확률을 바탕으로 하고 있습니다.   

Gensim은 bigram을 두 단어 사이에 '_'를 놓는 방식으로 표현합니다.   
bi-gram을 생성하는 코드는 다음과 같습니다.   

```
import gensim
bigram = gensim.models.Phrases(texts)
```

이제 학습된 bi-gram을 갖게 되었습니다.   
이 이후에 transformation을 하는 것은 TF-IDF와 동일합니다.   

```
texts = [bigram[line] for line in texts]

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]
```

bi-gram을 만든 후, tri-gram과 같은 다른 n-gram을 얼마든지 만들 수 있습니다.   
단지 phrases 모델을 여러번 돌리면 됩니다.   

### 전처리
모든 경우에 적용할 수 있는 전처리는 없습니다.   
data에 따라 적용해야 하는 전처리는 달라집니다.   

만약 너무 많이 나오는 단어와 너무 적게 나오는 단어를 제거하려 한다면 다음과 같은 기술을 적용할 수 있습니다.   

```
dictionary.filter_extremes(no_below=20, no_above=0.5)
```

20개의 문서보다 적게 나오거나, 50%의 문서 이상에서 나온 단어를 위와 같이 제거할 수 있습니다.   
