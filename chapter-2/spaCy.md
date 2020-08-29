## spaCy
---

### spaCy 소개
spaCy는 자기 자신을 **Industrial Strength Natural Language Processing**으로 소개하고 있습니다.   
spaCy는 POS tagging 알고리즘과 NER tagging 알고리즘에 집중하고 있습니다.   
이 말은, 다른 불필요한 기능들로 부풀려 있지 않다는 말입니다.   

spaCy는 academic approach에 집중하지 않습니다.   
이는 또 다른 NLP Library인 **NLTK**와 극명한 차이점을 보여줍니다.   

NLTK는 학생과 연구자들에게 다루어볼 만한 library를 제공하는 반면,   
spaCy는 실제 production code에서 사용될 수 있습니다.   
이 말은 실제 데이터를 적용해서 사용할 수 있다는 말입니다.   

spaCy의 특징은 다음과 같습니다.   

* Tokenization
* 21개 이상의 언어 지원
* 미리 학습된 word vectors
* Deep Learning과 쉬운 접목
* POS Tagging
* NER Tagging
* Dependency Parsing
* Sentence Segmentation
* Easy model packaging and deployment
* State-of-the-art speed

### spaCy 설치

spaCy를 직접 설치한다면 다음과 같이 설치할 수 있습니다.   

```
pip install -U spacy
```

하지만, 저는 virtual environments에서 설치하는 것을 권장합니다.   

```
virtualenv venv
source venv/bin/activate

pip install spacy
```

spaCy 설치를 마치고, import가 잘 되는지 테스트해보세요.   

```
import spacy
```