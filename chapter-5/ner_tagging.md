## NER Tagging
---

### NER Tagging이란?
NER은 Named Entity Recognition의 줄임말입니다.   
여기서 named entity가 무엇인지 알아봅시다.   
named entity는 특정한 이름을 가진 실제 세계의 객체입니다.   
예를 들면, France, Donald Trump, 그리고 Twitter같은 것이 있습니다.   
France는 GPE(Geopolitical Entity)로 분류되고, Donald Trump는 PER(person), Twitter는 ORG(organization)으로 분류됩니다.   

그렇다면 얼마나 많은 종류의 named entities가 존재할까요?
이것 역시 우리가 어떻게 정의하는가에 따라 다릅니다.   
우리가 기억해야 하는 것은 현대의 NER-tagger는 보통 통계적으로 학습된 모델이고, 문제에 따라 변할 수 있습니다.   

### spaCy에서 Entities의 종류
spaCy에서 사용하는 named entities의 카테고리는 다음과 같습니다.   

|타입|설명|
|:---|:-----------|
|PERSON|사람들|
|NORP|국적이나 종교적, 정치적 그룹|
|FACILITY|건물, 공항, 고속도로, 다리 등|
|ORG|회사, 법인 등|
|GPE|국가, 도시, 주 등|
|LOC|GPE가 아닌 locations, 산, 강 등|
|PRODUCT|사물, 운송수단, 음식 등|
|EVENT|허리케인, 전쟁 등의 사건|
|WORK_OF_ART|책 제목, 노래 등|
|LAW|법|
|LANGUAGE|언어|
|DATE|절대적, 상대적 날짜나 기간|
|TIME|하루보다 작은 단위의 시간|
|PERCENT|'%'가 포함된 퍼센트|
|MONEY|화폐|
|QUANTITY|무게나 거리 등의 크기|
|ORDINAL|서수|
|CARDINAL|다른 것과 중복되지 않는 숫자 타입|

### NER Tagging을 사용하는 이유
그렇다면 왜 NER Tagging을 사용하고, 왜 우리가 관심을 가져야 할까요?   
사실 단순히 named entities를 인지하는 것은 우리의 최종 목표는 아닙니다.   
하지만 NER Tagging은 우리가 최종 결과를 만들어내는데 하나의 중요한 주춧돌입니다.   
Entity linking은 어디에 entity recognition을 사용하고 그들 간의 관계를 알아내는 작업입니다.   

*Rome is the capital of Italy.*

NER tagger는 Rome을 place(GPE)로 인식하고, Italy 역시 마찬가지로 GPE로 인식할 것입니다.   
Rome을 도시로 인식하고 미국의 R&B artist로 인식하지 않는 것을 **Named Entity Disambiguation(NED)** 라고 부릅니다.   

NER tagger는 도메인에 따라 매우 다르게 인식되는 경향을 가지고 있습니다.   
POS tagger는 문서에 따라 약간의 차이를 보이지만, NER tagger는 문맥에 따라 완전히 다른 결과를 보여주기도 합니다.   
이런 현상은 매우 잘 훈련된 모델에서도 발생합니다.   
이는 어떤 도메인에서 사용하는지가 NER tagger에서는 매우 중요함을 알 수 있습니다.   

### NER Tagging을 사용하는 방법
POS tagger에서 dataset을 제공했던 것처럼, NER tagger에서도 학습을 위해서 문장과 NER tag를 제공합니다.   
또한 NER tagging을 할 때, 단어의 POS tag 역시 context로 사용합니다.   
이는 POS tagging을 NER tagging 전에 하는 이유 중 하나입니다.   
이 외에도 word의 prefix나 suffix가 어떤 특정한 symbol을 포함하는지, 이 것이 대문자인지 등이 NER tagging의 context로 사용됩니다.   

이렇게 features를 알게 되면, 우리 모델을 학습하는데 다수의 알고리즘이 사용됩니다.   
**CRF(Conditional Random Fields)** 가 NER tagging에 대부분 사용됩니다.   

물론, NLP에서 다른 작업들이 그렇듯, NER에서도 rule-based approch가 가능합니다.   
이 방법은 꽤 크고 방대한 용량의 dictionary를 필요로 하고, 지속적으로 업데이트되어야 하며, domain-specific하게 만들 수 있습니다.   
