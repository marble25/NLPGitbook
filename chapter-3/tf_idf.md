## TF-IDF
---

### TF-IDF란?
TF는 Term Frequency이고, IDF는 Inverse Term Frequency입니다.   
TF와 IDF를 수식으로 나타내면 다음과 같습니다.   

```
TF(t) = (t가 document에 나온 횟수) / (document의 총 단어 개수)
IDF(t) = log(document의 총 개수) / (t가 나온 document의 개수)
```

TF-IDF는 단순하게 TF와 IDF의 곱입니다.   
Vector 표현법에서 TF-IDF는 word의 개수를 세는 BoW 표현보다 더 많은 정보를 포함하고 있습니다.   
TF-IDF는 자주 나오지 않는 단어를 눈에 띄게 만들고, 너무 자주 나오는 단어의 중요도를 낮춥니다.   