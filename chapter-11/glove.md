## GloVe
---

GloVe는 Word2Vec처럼 context를 이용해서 단어를 vector로 표현하는 방법입니다.   
GloVe를 훈련하는 다양한 방법이 있지만, 이번에는 GloVe 사용에만 집중할 것입니다.   

미리 훈련된 GloVe는 [GloVe Download](http://nlp.stanford.edu/data/glove.6B.zip)에서 받을 수 있습니다.   

```
from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)
```

GloVe벡터를 로드한 후에 Word2Vec에서 했던 예제를 그대로 실행해 보겠습니다.   

```
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
```

결과는 다음과 같이 나옵니다.   

```
[('queen', 0.7698541283607483)]
```