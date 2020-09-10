## Document 살펴보기
---

### Document 다루기
topic 모델을 하나 가지게 되면, 우리는 어떤 주제들이 나왔는지 보는 것 뿐만 아니라, 문서들을 토픽에 따라 분류할 수 있습니다.   
이전 챕터에서 배웠던 코드를 통해 어떻게 topic이 생성되는지 알아봅시다.   

```
import os
import gensim

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

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.corpora import Dictionary

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

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)

print(ldamodel[corpus[0]])
```

결과는 다음과 같이 나올 것입니다.(값 자체는 조금 다를 수 있습니다)   

```
[(3, 0.9938318)]
```

이 tuple은 topic 번호와 해당 topic이 맞을 확률을 나타냅니다.   
이 list에 하나의 tuple만 있다는 것은, 다른 topic의 가능성은 무시할 만큼 작다는 이야기입니다.   

실제로 3번 토픽이 무엇인지 확인해 보겠습니다.   

```
print(ldamodel.show_topics()[3])
```

결과는 다음과 같습니다.   

```
(3, '0.004*"man" + 0.004*"Australia" + 0.004*"australian" + 0.004*"Arafat" + 0.003*"area" + 0.003*"Government" + 0.003*"claim" + 0.003*"come" + 0.003*"day" + 0.003*"fire"')
```

그렇다면 실제 text의 몇 문장을 확인해서 해당 topic의 단어가 있는지 봅시다.   

```
print(texts[0][:15])
```

결과는 다음과 같습니다.   

```
['hundred', 'people', 'force', 'vacate', 'home', 'Southern', 'Highlands', 'New_South', 'Wales', 'strong', 'wind', 'today', 'push', 'huge', 'bushfire']
```

topic에 나오는 단어들이 text에 들어가 있는 것을 볼 수 있습니다.   

하나 중요한 사실은 실행할 때마다 다른 topic, 다른 확률, 다른 단어들을 볼 수 있습니다.   
이는 topic 모델이 확률을 기반으로 하기 때문이고, 항상 실행할 때마다 다른 결과를 볼 수 있게 됩니다.   

다른 간단한 예시를 들어보겠습니다.   

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

ldamodel = LdaModel(corpus=corpus, num_topics=2, minimum_probability=1e-8, id2word=dictionary)
```

위의 corpus는 금융과 강, 두 개의 다른 토픽으로 이루어진 문장입니다.   
한 가지 알아야 할 점은 bank라는 단어가 두 문맥 모두 나올 수 있다는 점입니다.   

```
print(ldamodel.show_topics())
```

다음 코드를 실행시킨 결과입니다.   

```
[(0, '0.156*"water" + 0.150*"bank" + 0.142*"river" + 0.077*"tree" + 0.071*"flow" + 0.049*"mud" + 0.048*"shore" + 0.046*"fast" + 0.044*"rain" + 0.044*"finance"'), (1, '0.209*"bank" + 0.119*"sell" + 0.116*"money" + 0.101*"finance" + 0.083*"borrow" + 0.055*"water" + 0.050*"transaction" + 0.049*"loan" + 0.035*"flow" + 0.033*"river"')]
```

실제로는 다르게 나올 수 있습니다.   
하지만 한 topic은 river에 관련된 topic이고, 한 topic은 돈에 관련된 topic이라면 성공입니다.   
여기에서 다음 코드를 실행해 봅시다.   

```
print(ldamodel.get_term_topics('water'))
```

결과는 다음과 같은 형태로 나옵니다.(실제 결과값은 다를 수 있습니다)   

```
[(0, 0.14162402), (1, 0.03815759)]
```

0번 토픽에 더 많이 나온다는 것을 알 수 있습니다.   
이번에는 다음 코드를 실행해 보겠습니다.   

```
print(ldamodel.get_term_topics('finance'))
```

결과는 다음과 같이 나옵니다.   

```
[(0, 0.028981706), (1, 0.08443523)]
```

finance의 경우 1번 토픽에 더 많이 나오는 것을 알 수 있습니다.   
다음 코드를 실행해 봅시다.   

```
bow_water = ['bank', 'water', 'bank']
bow_finance = ['bank', 'finance', 'bank']

bow = ldamodel.id2word.doc2bow(bow_water)
doc_topics, word_topics, phi_values = ldamodel.get_document_topics(bow, per_word_topics=True)

print(word_topics)
```

다음과 같은 형태의 결과가 나옵니다.   

```
[(0, [0, 1]), (3, [0, 1])]
```

이 결과가 의미하는 바는 무엇일까요?   
tuple에서 첫 번째는 word의 index이고, 두 번째는 해당 word와 가까운 topic이 순서대로 나열되게 됩니다.   
0번 word인 bank와 3번 word인 water 모두 0번 topic(강물 관련)에 가깝게 나옵니다.   
`phi_values`에 대해서도 알아보겠습니다.   

```
print(phi_values)
```

결과는 다음과 같이 나옵니다.   

```
[(0, [(0, 1.2964872), (1, 0.70351094)]), (3, [(0, 0.9074108), (1, 0.09258791)])]
```

word_topics와 비슷하지만, phi_values는 해당 topic이 관련될 확률도 나옵니다.   
잘 보면 bank의 경우 bow에서 2번 나오기 때문에, phi_values의 합은 1이 아니라 2가 되게 됩니다.   
즉, 2배로 scaling되었다는 의미입니다.   

finance에 대해서도 같은 작업을 진행해 보겠습니다.   

```
bow = ldamodel.id2word.doc2bow(bow_finance)
doc_topics, word_topics, phi_values = ldamodel.get_document_topics(bow, per_word_topics=True)

print(word_topics)
```

결과는 다음과 같습니다.   

```
[(0, [1, 0]), (10, [1, 0])]
```

흥미롭게도, 이 경우에는 위에서 보았던 bank가 1번 토픽(finance 관련)에 더 관련성이 높다고 나옵니다.   
이 것은 finance라는 context 덕분입니다.   

지금까지 Gensim에서 LDA 모델을 다루는 방법을 알아보았습니다.   
같은 단어라도 문맥에 따라 다른 topic에 배정될 수 있다는 점도 알 수 있습니다.   