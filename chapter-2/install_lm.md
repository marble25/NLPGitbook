## Language Model 설치
---

### Language Model
spaCy의 특징 중 하나는 **Language Model**입니다.   
Language Model은 POS Tagging이나 NER Tagging과 같은 NLP 작업을 수행하는 통계적 모델입니다.   
이런 Language Model은 spaCy와 같이 패키지되어있지 않지만, download될 필요가 있습니다.   

다른 Language에는 다른 Language Model이 필요합니다.   
또한, 같은 Language에도 적용한 통계적 방법에 따른 다른 Language Model이 존재합니다.   
아쉽게도, 한국어에 대한 Language Model은 아직 존재하지 않습니다.   
그래서 앞으로 예제는 영어로 진행할 것 같습니다.   

### Language Model 설치
가장 쉽게 다운로드할 수 있는 방법은 spaCy의 `download` 커맨드를 이용하는 것입니다.   

```
spacy download en # english model
spacy download de # german model
spacy download fr # french model
spacy download zh # chinese model
spacy download xx # multi-language model
```

특정한 모델을 다운받는 법도 있습니다.   

```
spacy download en_core_web_sm
```

version까지 포함해서 다운받는 법도 있습니다.   

```
spacy download en_core_web_sm-2.0.0 --direct
```

다운로드가 완료되었으면 잘 작동하는지 테스트 해봅시다.   

```
import spacy

nlp = spacy.load('en')

doc = nlp(u'This is a sentence.')
```

만약 모델을 다운받았다면 다음과 같이 테스트할 수 있습니다.   

```
import spacy

nlp = spacy.load('en_core_web_sm')

doc = nlp(u'This is a sentence.')
```

### NLP 과정
Text가 주어졌을 때, NLP는 다음과 같은 순서로 Text에서 Doc으로 변환합니다.   

1. Tokenizer(토큰화)
1. Tensorizer(벡터화)
1. Tagger(POS)
1. Parser
1. Tagger(NER)

