## NER 태깅
---

### NER
Default Pipeline의 마지막은 NER(Named Entity Recognition)입니다.   
Named Entity는 이름이 부여된 실제 object입니다.(예를 들면, 사람, 나라, 제품, 회사 등)   
spaCy는 다양한 종류의 Named Entity를 모델에 예측하는 방식으로 인식할 수 있습니다.   

아래와 같이 사용할 수 있습니다.   
```
doc = nlp(u'Microsoft has offices all over Europe.')

for ent in doc.ents:
  print((ent.text, ent.start_char, ent.end_char, ent.label_))
```

그 결과는 아래와 같습니다.   
```
('Microsoft', 0, 9, 'ORG')
('Europe', 31, 37, 'LOC')
```

spaCy는 다음과 같은 built-in entity type을 가지고 있습니다.   
* PERSON: 사람
* NORP: 국적이나 정치적 그룹
* FACILITY: 빌딩, 공항, 도로, 다리 등
* ORG: 기관
* GPE: 나라, 도시, 주 등
* LOC: GPE가 아닌 location, 산, 강 등
* PRODUCT: 물건, 음식, 운송수단 등
* EVENT: 허리케인, 전투 등의 사건
* WORK_OF_ART: 책, 음악 등
* LAW: 법으로 만들어진 문서
* LANGUAGE: 언어


