## Gensim에서 벡터 표현법
---

### Gensim 설치
기구성된 virtual environment에서 다음과 같이 설치할 수 있습니다.   

```
pip install gensim
```

잘 설치되었는지 다음 코드로 확인해 보십시오.

```
from gensim import corpora

documents = [u'Football club Arsenal defeat local rivals this weekend.',
             u'Weekend football frenzy takes over London.',
             u'Bank open for takeover bids after losing millions.',
             u'London football clubs bid to move to Wembley stadium',
             u'Arsenal bid 50 million pounds for striker Kane.',
             u'Financial troubles result in loss of millions for bank.',
             u'Western bank files for bankruptcy after financial losses.',
             u'London football club is taken over by oil millionaire from Russia.',
             u'Banking on finances not working for Russia.']
```

위 코드가 잘 실행된다면, spaCy를 이용해서 전처리를 수행하십시오.

```
import spacy
nlp = spacy.load('en')
texts = []
for document in documents:
    text = []
    doc = nlp(document)
    for w in doc:
        if not w.is_stop and not w.is_punct and not w.like_num:
            text.append(w.lemma_.lower())
    texts.append(text)

print(texts)
```

결과는 다음과 같이 나올 것입니다.   

```
[['football', 'club', 'arsenal', 'defeat', 'local', 'rival', 'weekend'], ['weekend', 'football', 'frenzy', 'take', 'london'], ['bank', 'open', 'takeover', 'bid', 'lose', 'million'], ['london', 'football', 'club', 'bid', 'wembley', 'stadium'], ['arsenal', 'bid', 'pound', 'striker', 'kane'], ['financial', 'trouble', 'result', 'loss', 'million', 'bank'], ['western', 'bank', 'file', 'bankruptcy', 'financial', 'loss'], ['london', 'football', 'club', 'take', 'oil', 'millionaire', 'russia'], ['bank', 'finance', 'work', 'russia']]

```

이 corpus를 bag-of-words 방식으로 표현해 봅시다.   
Gensim의 *dictionary* class를 이용해서 편리하게 변환할 수 있습니다.   

```
dictionary = corpora.Dictionary(texts)
print(dictionary.token2id)
```

이 결과는 다음과 같습니다.   

```
{'arsenal': 0, 'club': 1, 'defeat': 2, 'football': 3, 'local': 4, 'rival': 5, 'weekend': 6, 'frenzy': 7, 'london': 8, 'take': 9, 'bank': 10, 'bid': 11, 'lose': 12, 'million': 13, 'open': 14, 'takeover': 15, 'stadium': 16, 'wembley': 17, 'kane': 18, 'pound': 19, 'striker': 20, 'financial': 21, 'loss': 22, 'result': 23, 'trouble': 24, 'bankruptcy': 25, 'file': 26, 'western': 27, 'millionaire': 28, 'oil': 29, 'russia': 30, 'finance': 31, 'work': 32}

```

corpus에 33개의 unique words가 존재하고, 각각의 단어는 dictonary에서 index를 할당받았습니다.   
이제 doc2bow method를 통해 document에서 BoW로 변환해 보겠습니다.   

```
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus)
```

corpus를 출력해 보면, BoW로 표현된 문서를 얻을 수 있습니다.   

```
[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)], [(3, 1), (6, 1), (7, 1), (8, 1), (9, 1)], [(10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1)], [(1, 1), (3, 1), (8, 1), (11, 1), (16, 1), (17, 1)], [(0, 1), (11, 1), (18, 1), (19, 1), (20, 1)], [(10, 1), (13, 1), (21, 1), (22, 1), (23, 1), (24, 1)], [(10, 1), (21, 1), (22, 1), (25, 1), (26, 1), (27, 1)], [(1, 1), (3, 1), (8, 1), (9, 1), (28, 1), (29, 1), (30, 1)], [(10, 1), (30, 1), (31, 1), (32, 1)]]
```

앞에서 BoW에 대해 설명한 것과 다르게, 1인 단어(있는 단어)는 (index, 개수)의 tuple의 형태로 list에 들어가게 되고, 0인 단어(없는 단어)는 아예 list에 없습니다.   

BoW에서 TF-IDF 형태로 변경하는 것 역시 Gensim에서 굉장히 쉽습니다.   

```
from gensim import models
tfidf = models.TfidfModel(corpus)

for document in tfidf[corpus]:
    print(document)
```

이 코드의 의미는 TF-IDF 모델을 학습한다는 의미입니다.   
다른 모델, LSA(Latent Semantic Analsis)나 LDA(Latent Dirichlet Allocation)의 경우는 더 복잡하고, 더 많은 시간이 필요하게 됩니다.   
이 코드의 결과는 다음과 같습니다.   

```
[(0, 0.3292179861221233), (1, 0.24046829370585296), (2, 0.4809365874117059), (3, 0.1774993848325406), (4, 0.4809365874117059), (5, 0.4809365874117059), (6, 0.3292179861221233)]
[(3, 0.24212967666975266), (6, 0.4490913847888623), (7, 0.6560530929079719), (8, 0.32802654645398593), (9, 0.4490913847888623)]
[(10, 0.18797844084016113), (11, 0.25466485399352906), (12, 0.5093297079870581), (13, 0.3486540744136096), (14, 0.5093297079870581), (15, 0.5093297079870581)]
[(1, 0.29431054749542984), (3, 0.21724253258131512), (8, 0.29431054749542984), (11, 0.29431054749542984), (16, 0.5886210949908597), (17, 0.5886210949908597)]
[(0, 0.354982288765831), (11, 0.25928712547209604), (18, 0.5185742509441921), (19, 0.5185742509441921), (20, 0.5185742509441921)]
[(10, 0.19610384738673725), (13, 0.3637247180792822), (21, 0.3637247180792822), (22, 0.3637247180792822), (23, 0.5313455887718271), (24, 0.5313455887718271)]
[(10, 0.18286519950508276), (21, 0.3391702611796705), (22, 0.3391702611796705), (25, 0.4954753228542582), (26, 0.4954753228542582), (27, 0.4954753228542582)]
[(1, 0.2645025265769199), (3, 0.1952400253294319), (8, 0.2645025265769199), (9, 0.3621225392416359), (28, 0.5290050531538398), (29, 0.5290050531538398), (30, 0.3621225392416359)]
[(10, 0.22867660961662029), (30, 0.4241392327204109), (31, 0.6196018558242014), (32, 0.6196018558242014)]
```

TF-IDF가 무엇인지 기억한다면, word_id 뒤에 나오는 float 값이 무엇인지 알 것입니다.   
단지 word count가 아닌, TF-IDF 값이 뒤에 나오게 됩니다.   