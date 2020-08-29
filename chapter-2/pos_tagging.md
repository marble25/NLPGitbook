## POS 태깅
---

### Tensorizer
Default Pipeline에서 두 번째 요소는 tensorizer(텍스트 벡터화)입니다.   
텍스트 벡터화는 doc을 내부적으로 float 배열 형태로 바꿉니다.   
하지만 spaCy가 내부적으로 알아서 처리하기 때문에, 우리는 이에 대해 걱정할 필요가 없습니다.   

### POS-tagging
이 예시를 보십시오.   
```
doc = nlp(u'John and I  went to the park')

for token in doc:
  print((token.text, token.pos_))
```

이 예시는 다음과 같은 결과를 가져옵니다.   
```
('John', 'PROPN')
('and', 'CCONJ')
('I', 'PRON')
('went', 'VERB')
('to', 'ADP')
('the', 'DET')
('park', 'NOUN')
('.', 'PUNCT')
```

POS Tagging이란 이와 같이 문법 요소를 token들에 할당하는 것을 말합니다.   

### Parsing
Default Pipeline에서 다음에 올 것은 parser로, dependency parsing을 수행합니다.   
Dependency Parsing은 symbol들 사이의 의존성에 대한 이해와 관련되어 있습니다.   
예를 들어, 영어에서는, 동사와 주어, 목적어 사이의 관계를 나타내는데 사용될 수 있습니다.   
