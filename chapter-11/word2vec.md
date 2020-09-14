## Word2Vec
---

Word2Vec 알고리즘은 굉장히 매력적이고 유용한 툴입니다.   
이 알고리즘은 단어를 벡터 형태의 표현으로 바꾸는 일을 합니다.   
*Efficient Estimation of Word Representations in Vector Space* 과 *Linguistic Regularities in Continuous Space Word* 라는 논문에서 Word2Vec의 초석을 다졌습니다.   

이 것이 정확히 어떤 일을 할까요?   
논문에서 든 예를 들어서 설명하자면, V(King) - V(Man) + V(Woman) = V(Queen), 즉 왕에서 남자 속성을 빼고 여자 속성을 더하면 여왕이 된다는 말입니다.   
이 것이 왜 주목을 받는지는 바로 여기에 담겨 있습니다.   
우리가 단어를 인지하는 것과 유사하게 벡터로 표현하게 됩니다.   

어떻게 Word2Vec이 동작하게 될까요?   
Word2Vec은 문맥을 이해하는데, 구체적으로 sliding window size를 정해서 그 단어들 사이에서 목표 단어가 나올 확률을 찾아냅니다.   
Word2Vec를 학습하는데는 **Continuous Bag of Words**(CBOW)와 Skip Gram model 메소드가 필요합니다.   

### Gensim에서 Word2Vec 사용하기

먼저 이 예제에서 사용할 파일을 다운로드 받겠습니다.   
[text8.zip](http://mattmahoney.net/dc/text8.zip)을 다운받아서 같은 폴더에 압축을 풀어줍니다.   
그 후, 예제를 진행합니다.   

```
from gensim.models import word2vec

sentences = word2vec.Text8Corpus('text8')
model = word2vec.Word2Vec(sentences, size=200, hs=1)
print(model)
```

꽤 오랜 시간을 기다린 후 결과는 다음과 같이 나옵니다.   

```
Word2Vec(vocab=71290, size=200, alpha=0.025)
```

이제 우리는 학습된 model을 가지게 되었습니다.   
유명한 King - Man + Woman 예제를 실행해 보겠습니다.   

```
print(model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
```

결과는 다음과 같이 나옵니다.   

```
[('queen', 0.5745400190353394)]
```

우리가 예상했듯이, queen이 결과로 나왔습니다.   
이 것은 역시 확률적인 과정이기 때문에 다른 단어를 얻을 가능성이 있지만, 대부분의 경우는 queen이 나올 것입니다.   
다음 코드를 실행해 봅시다.   

```
print(model.wv.most_similar(positive=['woman', 'king'], negative=['man']))
```

결과는 다음과 같이 나옵니다.   


```
[('queen', 0.5745400190353394),
 ('throne', 0.5453873872756958),
 ('regent', 0.5417712926864624),
 ('emperor', 0.532577395439148),
 ('empress', 0.5271527171134949),
 ('princess', 0.5225112438201904),
 ('consort', 0.5199817419052124),
 ('monarch', 0.5107275247573853),
 ('prince', 0.5055060982704163),
 ('sigismund', 0.48685187101364136)]
```

실제로 단어가 어떻게 벡터로 표현되었는지를 알려면 다음과 같이 입력하면 됩니다.   

```
print(model.wv['computer'])
```

size를 전에 200으로 설정했기 때문에 결과는 200차원의 벡터가 나오게 됩니다.   

`doesnt_match` 메소드를 통해 리스트에서 다른 단어들과 가장 멀리 떨어진 단어도 알아낼 수 있습니다.   

```
print(model.wv.doesnt_match('breakfast cereal dinner lunch'.split()))
```

결과는 'cereal'이 나오게 됩니다.   

또한, 얼마나 단어가 다른 단어와 가까운지도 알아낼 수 있습니다.   

```
print(model.wv.similarity('woman', 'man'))
print(model.wv.similarity('woman', 'cereal'))
print(model.wv.distance('woman', 'man'))
```

결과를 보면 'woman'과 'cereal'은 유사하지 않다고 알 수 있습니다.   

이미 학습된 구글 word2vec 모델을 다운로드받을 수 있습니다.   
그 후 다음과 같이 사용할 수 있습니다.   

```
from gensim.models import KeyedVectors

filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
```

이 모델은 300 차원으로 구성된 word vector입니다.   
이 모델을 실행하는 것은 앞의 예제와 유사하게 진행하면 됩니다.   