## 유사도 쿼리
---

이제 우리는 2개의 document를 비교할 수 있으므로, 가장 비슷한 document를 골라낼 수 있습니다.   
Gensim에서는 input query와 가장 비슷한 document를 찾아낼 built-in structures를 제공합니다.   

```
from gensim import similarities

index = similarities.MatrixSimilarity(model[corpus])
sims = index[lda_bow_finance]
```

`MatrixSimilarity`를 이용해서 corpus에서 Similarity를 계산할 index를 만들어 냈습니다.   
input query로는 `lda_bow_finance`를 사용했습니다.   

```
print(list(enumerate(sims)))
```

sims를 출력해 보면 다음과 같이 나옵니다.   

```
[(0, 0.35119006), (1, 0.24949455), (2, 0.30233026), (3, 0.3633883), (4, 0.2705042), (5, 0.99951863), (6, 0.99996334), (7, 0.9973844), (8, 0.99953634), (9, 0.9973635), (10, 0.99991035)]
```

document id와 query와 비슷한 정도가 출력되어 나옵니다.   
이를 정렬하여 texts와 매치해 보겠습니다.   

```
sims = sorted(enumerate(sims), key=lambda item: -item[1])

for doc_id, similarity in sims:
    print(texts[doc_id], similarity)
```

이 결과는 다음과 같습니다.   

```
['bank', 'borrow', 'money'] 0.99996334
['bank', 'loan', 'sell'] 0.99991035
['finance', 'money', 'sell', 'bank'] 0.99953634
['money', 'transaction', 'bank', 'finance'] 0.99951863
['bank', 'finance'] 0.9973844
['borrow', 'sell'] 0.9973635
['bank', 'bank', 'water', 'rain', 'river'] 0.3633883
['bank', 'river', 'shore', 'water'] 0.35119006
['bank', 'water', 'fall', 'flow'] 0.30233026
['river', 'water', 'mud', 'tree'] 0.2705042
['river', 'water', 'flow', 'fast', 'tree'] 0.24949455
```

돈에 관련된 문서들은 모두 높은 확률이 나오고, 강에 관련된 문서들은 모두 낮은 확률이 나오는 것을 알 수 있습니다.   