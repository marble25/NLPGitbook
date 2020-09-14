## Doc2Vec
---

Word2Vec를 만들면서, 연구자들은 Doc2Vec라고 불리는 paragraph나 document의 벡터 표현법을 만들어 냈습니다.   
Word2Vec과 유사하게 Doc2Vec는 2가지의 training method가 있습니다.   
CBOW와 Skip Gram의 변형인 PV-DM과 PV-DBOW를 사용합니다.   

다음 예시를 살펴보겠습니다.   

```
from gensim.models.doc2vec import TaggedDocument, LabeledSentence

sentence = TaggedDocument(words=['some', 'words', 'here'], tags=['SENT_1'])
```

다음과 같은 형식으로 태깅된 문서를 쉽게 만들 수 있습니다.   

실제로 lee corpus를 이용해서 모델을 한번 만들어 보겠습니다.   

```
from gensim.models.doc2vec import TaggedDocument, LabeledSentence
import os
import gensim

test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
lee_test_file = test_data_dir + os.sep + 'lee.cor'

def read_corpus(file_name, tokens_only=False):
    with open(file_name, encoding='iso-8859-1') as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                yield TaggedDocument(gensim.utils.simple_preprocess(line), [i])

train_corpus = list(read_corpus(lee_train_file))
test_corpus = list(read_corpus(lee_test_file, tokens_only=True))

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=100)
```

우리가 만든 모델은 50차원을 가지고, 2번보다 작게 나오는 단어를 제거한 후 100번의 반복을 통해 만들어지도록 세팅했습니다.   
실제로 train_corpus로 학습을 시켜보도록 하겠습니다.   

```
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
```

Doc2Vec를 훈련하는 방법은 2가지가 존재하는데, 다음과 같이 진행하면 됩니다.   

```
from gensim.models import Doc2Vec

moels = [
    # PV-DBOW
    Doc2Vec(dm=0, dbow_words=1, vector_size=200, window=8, min_count=10, epochs=50),
    # PV-DM
    Doc2Vec(dm=1, dm_mean=1, vector_size=100, window=8, min_count=10, epochs=50),
]
```

실제 실행은 다음과 같이 할 수 있습니다.   

```
inferred_vector = model.infer_vector(train_corpus[0].words)
sims = model.docvecs.most_similar([inferred_vector])
print(sims)
```

결과는 다음과 같이 나옵니다.   

```
[(0, 0.9918602705001831), (48, 0.8025969862937927), (40, 0.7799824476242065), (255, 0.732052743434906), (272, 0.7134027481079102), (33, 0.6928234100341797), (8, 0.6837693452835083), (19, 0.6445119976997375), (9, 0.6101011633872986), (264, 0.5773211121559143)]
```

0 document의 words와 가장 비슷한 것을 찾았는데 0 document를 제외하면 가장 비슷한 것은 48 document입니다.   
실제로 내용을 살펴보면 0번 document는 화재 발생과 소방관의 대응에 대한 내용입니다.   
그리고 48번 document 역시 소방관의 화재 진압에 대한 내용입니다.   
Doc2Vec를 통해 비슷한 document를 쉽게 찾아낼 수 있었습니다.   