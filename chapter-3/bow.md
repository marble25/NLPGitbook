## Bag-of-Words
---

### 벡터의 필요성
우리는 Text Analysis의 Machine Learning 파트로 들어왔습니다.   
이 말은 이제 word를 가지고 하기보다는 숫자들을 가지고 할 때가 왔다는 말입니다.   
spaCy를 사용할 때에도, POS tagging과 NER tagging에서 우리는 그저 unicode text를 입력으로 넣었을 뿐이지만, 내부적으로는 숫자를 사용했습니다.   

반면에, Gensim에서는 IR algorithm(LDA, LSI 등)의 input을 vector로 받기 때문에, string을 vector로 표현하는 것은 꼭 필요합니다.   

### Bag-of-Words
BOW 모델은 문장을 벡터로 표현하는 가장 직관적인 방법입니다.   
Chapter2에서 배운 방법으로 간단히 시작해 봅시다.   

```
import spacy

nlp = spacy.load('en')

doc1 = nlp(u'The dog sat by the mat.')
doc2 = nlp(u'The cat loves the dog.')
sentence1 = []
sentence2 = []
for w in doc1:
    if w.text != 'n' and not w.is_stop and not w.is_punct and not w.like_num:
        sentence1.append(w.text)

for w in doc2:
    if w.text != 'n' and not w.is_stop and not w.is_punct and not w.like_num:
        sentence2.append(w.text)

print(sentence1, sentence2)
```

S1은 "The dog sat by the mat."이고, S2는 "The cat loves the dog." 일 때, Stop words와 구두점, 숫자를 제거한 상태를 출력하는 코드입니다.   
결과는 다음과 같습니다.   

```
['dog', 'sat', 'mat'] ['cat', 'loves', 'dog']
```

이를 벡터로 표현하고 싶다면, 가장 먼저 vocabulary를 구축해야 합니다.   
우리의 vocabulary vector는 다음과 같습니다.   

```
vocab = ['dog', 'sat', 'mat', 'cat', 'loves']
```

중복된 단어인 'dog'는 제거되어 길이 5의 벡터로 변환했습니다.   
bag-of-words 모델은 다음과 같이 단어의 빈도를 포함합니다.   

```
S1:[1, 1, 1, 0, 0]
S2:[1, 0, 0, 1, 1]
```

이는 쉽게 이해할 수 있습니다.   
S1에는 'dog', 'sat', 'mat'이라는 단어가 있고, S2에는 'dog', 'cat', 'loves'라는 단어가 있기 때문에 해당되는 위치를 1로 표시하고, 나머지는 0으로 표시했습니다.   

BoW의 중요한 특징 중 하나는 순서가 없고 count만 있다는 점입니다.   
BoW 벡터만 가지고는 어떤 단어가 먼저 왔는지 알 수 없습니다.   
이는 위치 정보와 의미 정보를 잃어버릴 수 있습니다.   
하지만, 많은 Information Retrieval Algorithm에서 order가 중요하지 않기 때문에, 단어의 occurence만 가지고도 시작할 수 있습니다.   
예를 들어, spam filtering같은 곳에서 buy, money, stock과 같은 단어가 있다면 text를 BoW 벡터로 변환 후, Bayesian Probability를 적용해서 이메일이 spam인지 판단할 수 있습니다.   