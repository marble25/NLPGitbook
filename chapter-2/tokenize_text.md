## 텍스트 토큰화
---

### 토큰화란?
토큰화는 텍스트를 의미있는 segment인 token으로 나누는 것을 말합니다.   
이런 segment는 경우에 따라 word, 구두점, 숫자, 또는 다른 문자들이 될 수 있습니다.   
spaCy에서는 input이 unicode text가 되고, output은 doc object가 됩니다.   

각각의 언어는 각기 다른 tokenization rule을 가지고 있습니다.   
예시로, *Let us go to the park.*를 보겠습니다.   
각각의 token은 space 단위로 분리되게 됩니다.   
만약 이 문장이 *Let's go to the park.*였다면, tokenizer는 Let's를 Let과 's로 분리할 수 있어야 합니다.   
이 말은, tokenization에 띄어쓰기 외에도 특별한 규칙이 필요하다는 말입니다.   

Pipeline 내에 다른 부분과 다르게, tokenization에는 특별한 statistical model이 필요하지 않습니다.   
Tokenization은 prefix, suffix의 예외만 rule에 따라 분할해주는 방식입니다.   
Tokenization의 rule에 포함되지 않는 exception은 나중에 NER Tagging 등에서 처리를 해주게 됩니다.   