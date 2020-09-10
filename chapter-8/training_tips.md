## Training Tips
---

### 전처리
이전 챕터에서, 토픽 모델이 무엇인지에 대해 알아보았고, Gensim과 scikit-learn을 이용해서 어떻게 세팅하는지 알아보았습니다.   
그러나, 그냥 세팅만 하는 것은 충분하지 않습니다.   
학습이 잘 되지 않는다면, topic model은 우리에게 유용한 정보를 제공하지 않습니다.   
이를 위해, 앞에서도 언급했던 preprocessing, 전처리가 필요합니다.   

우리가 만든 topic 모델을 이용해서 처음 시도했을 때에는 좋은 결과가 나올 확률이 굉장히 적습니다.   
좋은 topic modeling을 위해서는 데이터를 정제하고, 결과를 읽은 후, 전처리 과정을 수정하는 수많은 작업들이 필요하기 때문입니다.   
예를 들어, 첫 topic model을 보고 stop word를 추가할 수 있습니다.   

spacy를 이용하면, 다음과 같이 stop word를 추가할 수 있습니다.   

```
import spacy
nlp = spacy.load('en')

my_stop_words = ['say', 'Mr', 'be', 'said', 'says', 'saying']
for stopword in my_stop_words:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True
```

다음 코드에서 `lexeme`의 `is_stop` 속성을 true로 변경했습니다.   
lexeme은 case sensitive하지 않으므로, case는 무시할 수 있습니다.   

NLTK에서는 다음과 같이 추가할 수 있습니다.   

```
from nltk.corpus import stopwords
stopword_list = stopwords.words('english')
```

`stopword_list`가 list 타입이기 때문에, 단지 여기에 새 단어를 추가하면 됩니다.   
혹시 stopwords가 없다면, 다음 코드를 먼저 실행하면 됩니다.   

```
import nltk
nltk.download('stopwords')

```

Stop word를 추가하는 것 외에도 Gensim의 Dictionary class를 이용하는 방법도 있습니다.   

```
from gensim.corpora import Dictionary
corpus = [['mama', 'male', 'maso'], ['ema', 'ma', 'mama']]
dct = Dictionary(corpus)
print(dct)

dct.filter_n_most_frequent(2)
print(dct)
```

위 코드의 결과는 다음과 같이 나옵니다.   

```
Dictionary(5 unique tokens: ['male', 'mama', 'maso', 'ema', 'ma'])
Dictionary(3 unique tokens: ['maso', 'ema', 'ma'])
```

처음 dictionary에서 `filter_n_most_frequent(2)` 함수를 호출하니, 가장 빈번히 나오는 단어 2개가 사라졌습니다.   
이 방식으로 자주 나오지만, 별로 중요한 의미가 없는 단어를 제거할 수 있습니다.   

이러한 작업들은 언제까지 해야할까요?
우리가 결과를 보았을 때 납득할 만한 정도가 될 때까지 해야 합니다.   
이 것이 '납득할 만한 정도'인지 판단하는 것은 *Topic Model 평가*에서 배울 것입니다.   

### Model Tuning
지금까지는 토픽 모델링 전에 해야할 것들을 알아보았습니다.   
이제 training에 넣을 options들을 수정하는 tuning 작업을 할 차례입니다.   
option들이 라이브러리에 따라 다르기는 하지만, 공통적으로 토픽의 개수는 꼭 들어가 있습니다.   

여기에는 진짜 정답이란 것은 없고, 이는 어떤 종류의 corpus인지, corpus의 크기 등 다양한 것들과 관련이 있습니다.   
만약 이에 대한 지식이 없다면, 5에서 시작해서 5씩 늘려가는 방법도 좋은 방법입니다.   

Gensim을 살펴보면, 중요한 tuning parameter는 다음과 같습니다.   

1. chunksize : 얼마나 많은 document가 한 번에 처리되는지. chunksize를 올리면 학습이 빨라집니다.   
1. passes : 전체 corpus를 몇 번 학습시킬 것인지. 다른 말로는 **epochs**라고도 합니다.   
1. iterations: 한 document를 몇 번씩 반복할 것인지. passes와 iterations를 적당히 높게 잡는 것이 중요합니다.   

Hyperparameters는 machine learning 이전에 설정하는 parameter를 말합니다.   
LDA model에서 2개의 hyperparameter가 있습니다.   

1. Alpha: document-topic 밀집도. 값이 높으면, document들이 더 많은 topic들로 구성되게 됩니다.   
1. Beta: topic-word 밀집도. 값이 높으면, topic이 더 많은 words로 구성되게 됩니다.   

