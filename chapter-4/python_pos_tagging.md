## Python에서 POS Tagging
---

### NLTK에서 POS tagging
이번에도 역시 spaCy에서 POS Tagging을 배울 것입니다.   
하지만, 그 전에 spaCy의 rival, NLTK의 POS Tagging이 제공하는 기능을 살펴봅시다.   

NLTK를 설치하는 과정은 다음과 같습니다.   

```
pip install nltk
```

NLTK는 간단히 설치되었습니다.   
NLTK에서 사용할 몇 가지 파일을 더 다운로드 받아보겠습니다.   

```
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

NLTK를 사용할 준비가 다 되었습니다.   
실제로 sentence에서 tag를 추출해 보겠습니다.   

```
import nltk

text = nltk.word_tokenize("And now for something completely different")
print(nltk.pos_tag(text))
```

결과는 다음과 같습니다.   

```
[('And', 'CC'), ('now', 'RB'), ('for', 'IN'), ('something', 'NN'), ('completely', 'RB'), ('different', 'JJ')]
```

만약 특정 tagger를 원한다면, 해당 tagger를 import해야 합니다.   
*train_sents*는 bigram tagger에 제공하는 training 문장들입니다.   

```
bigram_tagger = nltk.BigramTagger(train_sents)
bigram_tagger.tag(text)
```

### spaCy에서 POS tagging
POS tagging은 spaCy에서 제공하는 중요 기능 중 하나입니다.   
POS tagging을 해 봅시다.   

```
import spacy
nlp = spacy.load('en')

sent_0 = nlp(u'Mathieu and I went to the park.')
sent_1 = nlp(u'If Clement was asked to take out the garbage, he would refuse.')
sent_2 = nlp(u'Baptiste was in charge of refuse treatment center.')
sent_3 = nlp(u'Marie took out her rather suspicious and fishy cat to fish for fish.')

for token in sent_0:
    print((token.text, token.pos_, token.tag_))
```

결과는 다음과 같이 나올 것입니다.   

```
('Mathieu', 'PROPN', 'NNP')
('and', 'CCONJ', 'CC')
('I', 'PRON', 'PRP')
('went', 'VERB', 'VBD')
('to', 'ADP', 'IN')
('the', 'DET', 'DT')
('park', 'NOUN', 'NN')
('.', 'PUNCT', '.')
```

몇 개의 태그를 봅시다.   

Mathieu는 이름이고, PROPN(고유 명사)로 태그되었습니다.   
went는 VERB(동사)로, park는 NOUN(명사)으로 태그되었습니다.   

다음 코드를 실행해 봅시다.   

```
for token in sent_1:
    print((token.text, token.pos_, token.tag_))
```

결과는 다음과 같이 나옵니다.   

```
('If', 'SCONJ', 'IN')
('Clement', 'PROPN', 'NNP')
('was', 'AUX', 'VBD')
('asked', 'VERB', 'VBN')
('to', 'PART', 'TO')
('take', 'VERB', 'VB')
('out', 'ADP', 'RP')
('the', 'DET', 'DT')
('garbage', 'NOUN', 'NN')
(',', 'PUNCT', ',')
('he', 'PRON', 'PRP')
('would', 'VERB', 'MD')
('refuse', 'VERB', 'VB')
('.', 'PUNCT', '.')
```

여기서 refuse는 verb로 태그되었습니다.   

다음 코드를 실행해 봅시다.     

```
for token in sent_2:
    print((token.text, token.pos_, token.tag_))
```

결과는 다음과 같습니다.   

```
('Baptiste', 'PROPN', 'NNP')
('was', 'AUX', 'VBD')
('in', 'ADP', 'IN')
('charge', 'NOUN', 'NN')
('of', 'ADP', 'IN')
('refuse', 'NOUN', 'NN')
('treatment', 'NOUN', 'NN')
('center', 'PROPN', 'NNP')
('.', 'PUNCT', '.')
```

우리가 기대하던 결과가 나왔습니다.   
여기서 refuse는 noun으로 태그되었습니다.   

다음 코드를 실행해 봅시다.     

```
for token in sent_3:
    print((token.text, token.pos_, token.tag_))
```

결과는 다음과 같습니다.   

```
('Marie', 'PROPN', 'NNP')
('took', 'VERB', 'VBD')
('out', 'ADP', 'RP')
('her', 'PRON', 'PRP')
('rather', 'ADV', 'RB')
('suspicious', 'ADJ', 'JJ')
('and', 'CCONJ', 'CC')
('fishy', 'ADJ', 'JJ')
('cat', 'NOUN', 'NN')
('to', 'PART', 'TO')
('fish', 'VERB', 'VB')
('for', 'ADP', 'IN')
('fish', 'NOUN', 'NN')
('.', 'PUNCT', '.')
```

이 문장의 앞의 fish는 verb로, 뒤의 fish는 noun으로 올바르게 tag되었습니다.   