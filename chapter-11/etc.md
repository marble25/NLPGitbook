## 기타
---

이 외에도 다양한 Vectorization 메소드가 있습니다.   

### FastText

FastText는 Facebook AI research에서 개발한 Word2Vec를 확장한 벡터 표현법입니다.   
이름에서 드러나 있듯이, 빠르고, 같은 작업을 할 때 굉장히 효율적입니다.   
또한, 제한된 단어를 가지고도 똑똑한 word embeddings를 만들 수 있습니다.   
예륻 들어, embedding(strange) - embedding(strangely) ~= embedding(charming) - embedding(charmingly)입니다.   

### WordRank

이름에서 제시하듯, embeddings를 ranking problem으로 풀려고 합니다.   
기본적인 idea는 GloVe와 비슷합니다.   

### Varembed

FastText처럼 이는 형태학적인 정보를 활용하는데 장점이 있습니다.   
GloVe vector처럼, 우리는 모델을 새 단어로 업데이트할 수 없고, 새 모델을 학습해야만 합니다.   

### Poincare

Poincare는 Facebook AI research에서 개발한 embedding입니다.   
핵심 아이디어는 단어 사이의 관계를 이해하고 단어 embedding을 만드는데 시각적 표현법을 사용하는 것입니다.   
