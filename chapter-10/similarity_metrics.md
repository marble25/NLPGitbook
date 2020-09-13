## 유사도 행렬
---

유사도 행렬은 nlp에서 자주 사용되는 수학적 구조입니다.   
유사도 행렬 중 하나인 Euclidean metric에 대해서 배웠을 것입니다.   
유사도 행렬은 다음과 같은 특징이 있습니다.   

* d(x, y) >= 0
* d(x, y) = 0 <=> x = y
* d(x, y) = d(y, x)
* d(x, z) <= d(x, y) + d(y, z)

Gensim에서는 이러한 distance metric을 package 안에 포함시켜서 document와 topic 차원에서 쉽게 사용할 수 있도록 했습니다.   

Chapter-8에서 살펴보았던 코드를 사용해서 하도록 하겠습니다.   

```
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.corpora import Dictionary

texts = [['bank', 'river', 'shore', 'water'],
         ['river', 'water', 'flow', 'fast', 'tree'],
         ['bank', 'water', 'fall', 'flow'],
         ['bank', 'bank', 'water', 'rain', 'river'],
         ['river', 'water', 'mud', 'tree'],
         ['money', 'transaction', 'bank', 'finance'],
         ['bank', 'borrow', 'money'],
         ['bank', 'finance'],
         ['finance', 'money', 'sell', 'bank'],
         ['borrow', 'sell'],
         ['bank', 'loan', 'sell']]

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
```

text를 만들고, bow 형태의 corpus도 만들어 주었습니다.   
TF-IDF 형태로 변환하고, LDA 모델도 만들어 주겠습니다.   

```
from gensim.models import ldamodel
from gensim.models import TfidfModel

tfidf = TfidfModel(corpus)
model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=2)

print(model.show_topics())
```

토픽이 어떻게 나오는지 보겠습니다.   

```
[(0, '0.202*"bank" + 0.113*"money" + 0.105*"finance" + 0.094*"sell" + 0.077*"borrow" + 0.065*"water" + 0.050*"river" + 0.048*"transaction" + 0.042*"flow" + 0.039*"shore"'), (1, '0.155*"bank" + 0.151*"water" + 0.129*"river" + 0.083*"tree" + 0.065*"flow" + 0.051*"rain" + 0.050*"fast" + 0.048*"mud" + 0.048*"sell" + 0.037*"finance"')]
```

비교할 3개의 도큐먼트를 만들고(하나는 강에 대한 것, 하나는 돈에 대한 것, 하나는 둘 다 가지고 있는 것) 이를 bow로 변환하고, TF-IDF와 Lda로 바꿔보겠습니다.   

```
doc_water = ['river', 'water', 'shore']
doc_finance = ['finance', 'money', 'sell']
doc_bank = ['finance', 'bank', 'tree', 'water']

bow_water = model.id2word.doc2bow(doc_water)
bow_finance = model.id2word.doc2bow(doc_finance)
bow_bank = model.id2word.doc2bow(doc_bank)

lda_bow_water = model[bow_water]
lda_bow_finance = model[bow_finance]
lda_bow_bank = model[bow_bank]

tfidf_bow_water = tfidf[bow_water]
tfidf_bow_finance = tfidf[bow_finance]
tfidf_bow_bank = tfidf[bow_bank]
```

`lda_bow_water`를 보면 다음과 같은 값이 나옵니다.   

```
[(0, 0.8464617), (1, 0.15353829)]
```

이 document는 강에 관련된 단어를 더 많이 가지고 있으므로, 강에 대한 topic이 훨씬 높게 나오는 것이 당연합니다.   
`lda_bow_finance`를 보면 다음과 같은 값이 나옵니다.   

```
[(0, 0.13444014), (1, 0.8655599)]
```

이 document는 돈에 관련된 단어를 더 많이 가지고 있으므로, 돈에 대한 topic이 훨씬 높게 나옵니다.   
`lda_bow_bank`를 보면 다음 값이 나옵니다.   

```
[(0, 0.5956138), (1, 0.4043862)]
```

강에 대한 단어가 더 많이 있다고는 나오지만, 그 차이가 크지는 않다고 나옵니다.   

Gensim에는 **Hellinger metric**, **Kullback-Leiber divergence function**, **Jaccard index**라는 거리 함수가 포함되어 있습니다.   
두 개의 document를 비교할 때에 완벽한 거리 함수는 없기 때문에 잘 비교해 보면서 사용하는 것이 필요합니다.   

```
from gensim.matutils import kullback_leibler, jaccard, hellinger

print(hellinger(lda_bow_water, lda_bow_finance))
print(hellinger(lda_bow_finance, lda_bow_bank))
print(hellinger(lda_bow_water, lda_bow_bank))
```

결과는 다음과 같이 나옵니다.   

```
0.5459944566444418
0.3541193455521899
0.20193702755596438
```

water과 finance 사이의 거리가 가장 먼 것으로 나오고, finance와 bank, water과 bank는 그에 비하면 비교적 가까운 것으로 나옵니다.   
0에 가까울 수록 비슷한 문서이고, 1에 가까울 수록 차이가 많은 문서입니다.   

Kullback-Leiber Function에 대해 기억할 중요한 사실이 있습니다.   
정확히 말하면 이 function은 거리 함수가 아닌데, 그 이유는 symmetric하지 않기 때문입니다.   
다시 말하자면, `kullback_leibler(lda_bow_bank, lda_bow_finance)`와 `kullback_leibler(lda_bow_finance, lda_bow_bank)`가 동일하지 않다는 이야기입니다.   
이 둘의 결과는 다음과 같이 다릅니다.   

```
0.57881486
0.45858586
```

그 반면에 `hellinger(lda_bow_bank, lda_bow_finance)`와 `hellinger(lda_bow_finance, lda_bow_bank)`는 다음과 같이 동일합니다.   

```
0.3541193455521899
0.3541193455521899
```

KL function은 완전히 거리 함수의 성질을 갖지는 않지만 0에 가까울 수록 비슷하고, 1에 가까울 수록 다르다는 성질은 가지고 있습니다.   

마지막 함수는 Jaccard metric으로, 이 함수는 다른 2개의 함수와 다르게 bow와 doc에 대해서도 동작합니다.   

```
print(jaccard(bow_water, bow_bank))
print(jaccard(doc_water, doc_bank))
print(jaccard(['word'], ['word']))
```

결과는 다음과 같습니다.   

```
0.8571428571428572
0.8333333333333334
0.0
```

이 함수 역시 0에 가까울 수록 비슷하고, 1에 가까울 수록 차이가 큽니다.   

이제 topic들 사이에 얼마나 차이가 있는지 알아보는 작업을 해 보겠습니다.   

```
def make_topics_bow(topic):
    topic = topic.split('+')
    topic_bow = []
    for word in topic:
        prob, word = word.split('*')
        word = word.replace(' ', '').replace('"', '')
        word = model.id2word.doc2bow([word])[0][0]
        topic_bow.append((word, float(prob)))
    return topic_bow

topic_water, topic_finance = model.show_topics()
finance_distribution = make_topics_bow(topic_finance[1])
water_distribution = make_topics_bow(topic_water[1])
```

topic들을 word id와 할당된 몫의 형태로 변환해 보았습니다.   
`finance_distribution`을 출력해 보면 다음과 같습니다.   

```
[(0, 0.232),
 (10, 0.111),
 (11, 0.111),
 (14, 0.1),
 (13, 0.072),
 (1, 0.053),
 (12, 0.048),
 (15, 0.046),
 (3, 0.043),
 (2, 0.036)]
```

이제 `hellinger`함수로 두 개의 distribution을 비교해 보겠습니다.   

```
print(hellinger(water_distribution, finance_distribution))
```

결과는 다음과 같습니다.   

```
0.6564931577500894
```