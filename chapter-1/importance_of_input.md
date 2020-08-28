## Garbage in, Garbage out
---

### Garbage in, Garbage out이란?
Garbage in, Garbage out(GIGO)는 컴퓨터 과학에 사용되는 용어 중 하나입니다.   
우리가 이상한 데이터를 넣는다면, 높은 확률로 이상한 결과를 얻을 것이라는 말입니다.   

많은 데이터는 더 나은 예측을 가능하게 하지만, 언제나 옳은 것은 아닙니다.   
직관적인 예를 들자면, *a*나 *the*같은 단어는 text에서 많이 등장하지만, text에 아무런 정보를 더해주지 않습니다.   

이와 같이, 우리에게 유용한 정보를 주지 않는 단어들을 **stop words**라고 하고, 이런 단어들은 보통 text analysis가 이루어지기 전에 text에서 제거됩니다.   
비슷하게, 우리는 text에서 굉장히 많이 나오거나 굉장히 적게(1~2번) 나오는 단어를 제거하는 경우가 있습니다.   
하지만, task에 따라, stop words를 제거하지 않아야 될 때도 있습니다.   
stop words도 유용한 정보를 포함할 수 있기 때문이죠.(자세한 내용은 *Pastiche detection based on stopword rankings. Exposing impersonators of a Romanian writer* 참고)   

필요 없는 데이터를 제거해야하는 또 다른 경우를 봅시다.   
text에서 topic을 찾는다고 할 때, reading과 read를 따로 취급하는것이 맞을까요?   
reading을 read로 바꾼다고 해도, 사라지는 정보가 없기 때문에 reading을 read로 변환하는 것이 현명합니다.   
반대로, information과 inform은요?   
information과 inform은 context에 따라 다른 의미를 가질 수 있기 때문에 변환하지 않는 것이 현명합니다.   
이런 Lemmatizing과 Stemming은 NLP에서 중요한 개념 중 하나입니다.   

이런 기본적인 Text-Processing을 거친 후에도, 데이터는 여전히 단어의 모임입니다.   
우리는 이를 숫자 형태로 변환해 주어야 합니다.   
이를 위해 **Bag-of-words(BOW)**나 **Term Frequency-Inverse Document Frequency(TF-IDF)** 등이 사용됩니다.   
이 외에도 **Word2Vec**나 **GloVe** 같은 고급 테크닉이 많이 존재합니다.   