## 텍스트 전처리
---

### Stop words
Stop words(불용어)는 어떤 정보도 제공해주지 않는 말을 의미합니다.   
Stop words는 NLP algorithm을 적용하기 전에 제거됩니다.   
다시 말하지만, 모든 경우에 적용되는 것은 아닙니다.   

spaCy에서는 stop words는 굉장히 알기 쉽습니다.   
is_stop이라는 속성을 가지고 있는 token은 stop word입니다.   

우리의 stop word를 추가해 봅시다.   
```
my_stop_words = [u'say', u'be', u'said', u'says', u'saying']
for stopword in my_stop_words:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True
```

기본적인 stop word에는 어떤 것이 있는지 확인해 봅시다.   
```
from spacy.lang.en.stop_words import STOP_WORDS
print(STOP_WORDS)
```

### Stemming과 Lemmatization
saying과 say, says는 문법적인 차이만 있을 뿐, 전달하는 의미는 동일합니다.   
이런 단어들을 처리하는 방법에는 stemming과 lemmatization이 있습니다.   

Stemming은 단어의 끝을 지우는 방식입니다.   
예를 들어, saying과 say, says는 모두 say가 됩니다.   

Lemmatization은 형태학적 분석을 통해 root word를 찾아가는 것입니다.   
예를 들어, is와 are, were는 모두 'be'라는 root word를 가지고 있습니다.   

실제로, preprocessing을 진행해 봅시다.   
이 코드는 stop word를 추가한 이후 진행되는 코드입니다.   
```
doc = nlp(u'the horse galloped down the field and past the river. ')
sentence = []
for w in doc:
    if not w.is_stop and not w.is_punct and not w.like_num:
        sentence.append(w.lemma_)

print(sentence)
```

위에서 `is_stop`과 `is_punct`, `like_num`을 통해 불용어와 구두점, 숫자 표현을 제거했습니다.   
결과는 다음과 같습니다.   
```
['horse', 'gallop', 'past', 'river']
```
위 예시에서는 숫자를 중요하지 않은 정보로 취급했지만, 다른 경우에는 중요한 정보일 수 있습니다.   
