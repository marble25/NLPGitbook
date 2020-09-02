## Python에서 NER Tagging
---

### chunking
먼저 chunking이라는 단어에 대해 알아볼 필요가 있습니다.   
이는 POS Tagging이 끝난 후에 연속적인 부분으로 문장을 분할하는 것을 의미합니다.   

*The little brown dog barked at the black cat.*

이 경우에, 두 개의 명사구를 쉽게 찾아낼 수 있습니다.   
'The little brown dog'와 'the black cat'
이는 chapter 6에서 더 자세히 배울 것입니다.   

왜 이 것이 NER Tagging과 관련이 있을까요?
이전에 든 예시에서, 단지 Donald나 Trump가 아닌, Donald Trump라는 전체 명사구가 사람으로 태그되었습니다.   
명사구에 대한 이해는 tagging에서 우리의 결정을 도와줄 것입니다.   

온라인에서 tagging 시스템에 대해 찾아보면, IOB 시스템 같은 것이 나올 것입니다.   
spaCy에서는 이를 더 세분화해서 BILOU 시스템을 사용합니다.   

|태그|설명|
|:---|:------|
|`B`EGIN|멀티 토큰 객체의 첫 토큰|
|`I`N|멀티 토큰 객체의 내부 토큰|
|`L`AST|멀티 토큰 객체의 마지막 토큰|
|`U`NIT|단일 토큰 객체|
|`O`UT|Entity가 아닌 토큰|

### NLTK에서 NER Tagging
POS tagger에서 살펴보았듯이, NER Tagging에서도 spaCy의 최대 라이벌인 NLTK의 기능을 구현해 봅시다.   
먼저 몇 가지 설치해야 하는 것이 있습니다.   

```
import nltk

nltk.download('maxent_ne_chunker')
nltk.download('words')
```

그 후 다음 코드를 실행해봅시다.   

```
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk import pos_tag
from nltk import word_tokenize
from nltk.chunk import ne_chunk


sentence = "Clement and Mathieu are working at Apple."
ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))

iob_tagged = tree2conlltags(ne_tree)
print(iob_tagged)
```

결과는 다음과 같이 나왔습니다.   

```
[('Clement', 'NN', 'B-GPE'), ('and', 'CC', 'O'), ('Mathieu', 'NNP', 'B-PERSON'), ('are', 'VBP', 'O'), ('working', 'VBG', 'O'), ('at', 'IN', 'O'), ('Apple', 'NNP', 'B-ORGANIZATION'), ('.', '.', 'O')]
```

코드를 보면, 먼저 문장을 tokenize했고, POS-tag 했고, 그 후 chunk했습니다.   
결과를 보면, Clement가 사람인데도 GPE로 잘못 나와 있고, 그 외에는 다 잘 나와 있습니다.   

다른 유명한 tagger는 Stanford NER tagger입니다.   
Stanford NER tagger는 앞서 살펴봤던 CRF 알고리즘을 사용해서 tagging합니다.   
이 tagger를 사용하기 위해서는 Jar File을 다운로드받아야 하는데, 다음 사이트에서 다운로드받을 수 있습니다.   
[https://nlp.stanford.edu/software/CRF-NER.html](https://nlp.stanford.edu/software/CRF-NER.html)

그 후 아래와 같이 잘 실행되는지 확인해 봅시다.   

```
from nltk.tag import StanfordNERTagger
st = StanfordNERTagger('stanford-ner-4.0.0/classifiers/english.all.3class.distsim.crf.ser.gz',
                       'stanford-ner-4.0.0/stanford-ner.jar', encoding='utf-8')

print(st.tag('Baptiste Capdeville is studying at Columbia University in NY'.split()))
```

결과는 다음과 같이 나옵니다.   

```
[('Baptiste', 'PERSON'), ('Capdeville', 'PERSON'), ('is', 'O'), ('studying', 'O'), ('at', 'O'), ('Columbia', 'ORGANIZATION'), ('University', 'ORGANIZATION'), ('in', 'O'), ('NY', 'O')]
```

NLTK는 간단하기는 하지만, 실제 production level에서 사용할 만한 수준은 아닙니다.   

### spaCy에서 NER Tagging
실제로 간단한 예시와 함께 NER Tagging을 진행해 봅시다.   

```
import spacy
nlp = spacy.load('en')

sent_0 = nlp(u'Donald Trump visited at the government headquarters in France today.')
sent_1 = nlp(u'Emmanuel Jean-Michel Frédéric Macron is a French politician serving as President of France and ex officio Co-Prince of Andorra since 14 May 2017.')
sent_2 = nlp(u"He studied philosophy at Paris Nanterre University, completed a Master's of Public Affairs at Sciences Po, and graduated from the École Nationale Administration (ÉNA) in 2004.")
sent_3 = nlp(u'He worked at the Inspcectorate General of Finances, and later became an investment banker at Rothschild & Cie Banque.')

for token in sent_0:
    print((token.text, token.ent_type_))
```

위 코드의 결과는 다음과 같습니다.   

```
('Donald', 'PERSON')
('Trump', 'PERSON')
('visited', '')
('at', '')
('the', '')
('government', '')
('headquarters', '')
('in', '')
('France', 'GPE')
('today', 'DATE')
('.', '')
```

Named Entities로 인식되지 않은 것들은 empty stringdmfh skdhrp ehlqslek.   
이 문장에는 Donald Trump, France, today라는 3개의 entity가 존재하는데, 각각 PERSON(사람), GPE(국가 그룹), DATE(날짜)로 잘 분류되었습니다.   
goverment headquarters의 경우 어떤 특정한 것을 가리키지 않기 때문에 named entity로 분류되지 않았습니다.   

spaCy가 `doc.ents`에 entities를 저장해놓는다는 것을 기억해 보세요.   

```
for ent in sent_0.ents:
    print((ent.text, ent.label_))
```

위 코드의 결과는 다음과 같이 나올 것입니다.   

```
('Donald Trump', 'PERSON')
('France', 'GPE')
('today', 'DATE')
```

결과를 보면, Donald Trump가 하나의 entity로 취급된다는 것에 주목하십시오.   

다음 문장을 보겠습니다.   

```
for ent in sent_1.ents:
    print((ent.text, ent.label_))
```

결과는 다음과 같이 나옵니다.   

```
('Emmanuel Jean-Michel', 'PERSON')
('Frédéric Macron', 'PERSON')
('French', 'NORP')
('France', 'GPE')
('Andorra', 'PERSON')
('14 May 2017', 'DATE')
```

여기에서 몇 가지 오류가 보입니다.   
Emmanuel Jean-Michel Frédéric Macron 전체가 사람이고, Andorra는 국가로 취급되어야 합니다.   

다음 문장을 보겠습니다.   

```
for ent in sent_2.ents:
    print((ent.text, ent.label_))
```

결과는 다음과 같이 나옵니다.   

```
('Paris Nanterre University', 'ORG')
('a Masters of Public Affairs at Sciences Po', 'WORK_OF_ART')
('the École Nationale Administration', 'ORG')
('ÉNA', 'ORG')
('2004', 'DATE')
```

꽤 잘 나오는 것처럼 보입니다.   

다음 장에서는 spaCy의 NER model 학습을 배워보겠습니다.   