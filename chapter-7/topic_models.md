## Topic Models
---

### Topic Model이란?
우리는 지금까지 POS tagging과 NER tagging, Dependency tagging에 대해 배우면서 통계적 모델에 대해 배웠습니다.   
하지만 우리의 목표는 단지 통계적 모델을 구축하는 것이 아닙니다.   

Topic Model이 무엇일까요?   
그 이름이 제시하듯, 텍스트의 주제에 대한 정보를 제공해주는 확률적인 모델을 말합니다.   

이런 토픽 모델이 왜 중요할까요?   
전통적으로, 정보 획득이나 검색 테크닉의 경우에 단어들 사이의 유사성과 관련성을 인지하는 것을 포함하고 있습니다.   
그래서 단지 단어들보다는, topic을 통해 파일을 더 잘 찾고, 분류할 수 있게 됩니다.   

토픽이 정확히 무엇인지 아는 것이 중요합니다.   
토픽은 단어의 확률적인 분포를 의미합니다.   
우리는 이 모델을 사용해서 문서를 topic의 확률적인 분포로 나타낼 수 있습니다.   
우리가 단어와 단어의 개수를 알고 있기 때문에, 이 정보를 토픽 모델을 만드는데 사용할 수 있습니다.   

우리가 알고 있어야 하는 것은, topic은 단지 단어의 확률적인 분포이기 때문에, 그들만의 title이나 라벨을 만들지 않는다는 것입니다.   
예를 들어, 우리가 신문에서 weather topic이라고 부르는 것은, 단지 단어의 모음일 것이고(해, 온도, 바람, 예보 등), 이 단어들 사이의 연관 확률로 표현될 것입니다.   

이런 topic model을 만드는 방법은 한 가지가 아니고, 이들을 만드는 방법에 대해 Gensim을 이용ㅇ해서 배워볼 것입니다.   
LDA, LSA, HDP, DTM 등의 방법은 공통점이 있습니다.   
바로 단어들이 확률적인 분포 아래 있다고 가정하고 이 분포를 찾아내려고 시도한다는 것입니다.   

### Gensim에서 Topic Model
Gensim은 무료로 사용할 수 있는 가장 유명한 topic modeling toolkit입니다.   
Gensim은 다양한 topic modeling algorithm을 사용할 수 있고, 직관적이고, 커뮤니티가 활성화되어 있습니다.   

Gensim으로 전처리 작업을 진행해 보겠습니다.   

```
import matplotlib.pyplot as plt
import gensim
import numpy as np
import spacy

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary

import os, re, operator

test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
text = open(lee_train_file).read()

import spacy
nlp = spacy.load("en")

my_stop_words = [u'say', u'\'s', u'Mr', u'be', u'said', u'says', u'saying']
for stopword in my_stop_words:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True

doc = nlp(text)
print(doc)
```

다음을 실행하면 매우 긴 문장들이 300줄에 걸쳐서 나올 것입니다.   
이전 단원들에서 공부했던 내용들을 되짚어 보면서 다음 코드를 실행해 보겠습니다.   

```
# we add some words to the stop word list
texts, article, skl_texts = [], [], []
for w in doc:
    # if it's not a stop word or punctuation mark, add it to our article!
    if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num:
        # we add the lematized version of the word
        article.append(w.lemma_)
    # if it's a new line, it means we're onto our next document
    if w.text == '\n':
        skl_texts.append(' '.join(article))
        texts.append(article)
        article = []

bigram = gensim.models.Phrases(texts)
texts = [bigram[line] for line in texts]
print(texts[1][0:10])

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

print(corpus[1][0:10])
```

위 코드를 살펴보면, doc에서 stop word와 구두점, 숫자 등을 제거한 후 단어에 stemming을 하여 texts에 추가해 주었습니다.   
bigram을 만들어 주고, 이 bigram을 이용해서 bow 형태로 변환해서 corpus에 저장해 줍니다.   
결과는 다음과 같이 나옵니다.   

```
['indian', 'security_force', 'shoot_dead', 'suspect', 'militant', 'night', 'long', 'encounter', 'southern', 'Kashmir']
[(71, 1), (83, 1), (91, 1), (93, 1), (94, 1), (109, 1), (110, 1), (111, 1), (112, 4), (113, 1)]
```

이로써 간단히 Gensim에서 Topic Model 사용할 준비를 마쳤습니다.   