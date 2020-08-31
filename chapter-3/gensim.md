## Gensim
---

### Gensim이란?
현재까지 우리는 spaCy를 이용하면서 숨겨진 정보를 알아내는 것보다는 text의 shape을 맞추는데 주력해 왔습니다.   
이 말은 우리가 아직 Preprocessing 단계에 있다는 것을 말합니다.   
이번 단원에서 string들을 vector로 표현하는 방법들에 대해서 배워볼 것입니다.   
이런 방법에는 BOW(Bag of Words), TF-IDF(Term Frequency-Inverse Document Frequency), LSI(Latent Semantic Indexing), Word2Vec 등이 있습니다.   
이런 Transformed Vector들은 scikit-learn machine에 쉽게 삽입될 수 있습니다.   

Gensim은 python의 built-in generator과 iterator를 이용하기 때문에 scalable할 수 있고, dataset은 RAM에 전부 로드되지 않게 됩니다.   
Gensim은 memory와 독립적이고, 프로세스 처리에 multicore를 사용 가능합니다.   

