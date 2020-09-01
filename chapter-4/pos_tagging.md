## POS Tagging
---

### POS Tagging이란?
POS는 Part-Of-Speech의 줄임말로, 한국어로 품사를 의미합니다.   
이 말은, POS Tagging이란 입력으로 들어온 단어들의 품사를 지정해주는 작업이라는 의미입니다.   

전통적으로, POS는 문법적인 특징 및 사용과 밀접한 연관성이 있습니다.   
영어의 품사를 분류할 때 흔히 사용되는 카테고리는 다음과 같습니다.   

* Noun : 명사 (사람, 사물, 장소 등)
* Verb : 동사 (움직임이나 상태)
* Adjective : 형용사 (명사나 대명사를 꾸며주거나 설명함)
* Adverb : 부사 (동사나 형용사, 다른 부사를 꾸며주거나 설명함)
* Pronoun : 대명사 (명사 대신에 사용되며, 명사를 지칭함)
* Preposition : 전치사 (명사나 대명사 앞에 위치하여 전치사구(phrase)를 형성하고, 전치사구는 문장 내에 다른 단어들을 수식하는데 사용됨)
* Conjunction : 접속사 (단어, 구, 절 등을 연결해줌)
* Interjection : 감탄사 (감정을 표현함)

이 외에도 다양한 카테고리가 존재할 수 있습니다.   

만약 영어 이외에 다른 언어를 처리할 때에는 다른 POS tagger를 사용해야 합니다.   
영어와는 분류하는 기준이 다를 수 있고, 카테고리 자체도 달라질 수 있기 때문입니다.   
심지어 영어에서도, POS tagging은 항상 직관적인 것은 아닙니다.   
예를 들어, 단어 *refuse*의 경우 명사로 사용되면 '쓰레기'를 의미하지만, 동사로 사용되면 '거절하다'를 의미합니다.   
이렇듯 문장에서 어떤 의미로 사용되는지를 아는 것은 굉장히 중요합니다.   
POS Tag를 알아내려면 문맥이 필수적입니다.   

spaCy에서는 조금 더 디테일한 분석을 진행합니다.   
spaCy에서 사용하는 카테고리는 다음과 같습니다.   

|태그|설명|예시|
|:---|:------|:----------------|
|ADJ| adjective(형용사) |big, old, green, incomprehensible, first|
|ADP| adposition(부치사) |in, to, during|
|ADV| adverb(부사) |very, tomorrow, down, where, there|
|AUX| auxiliary(보조사) |is, has, willl, should|
|CONJ| conjunction(접속사) |and, or, but|
|CCONJ| coordinating conjunction(등위 접속사) | and, or, but |
|DET| determiner(한정사) |a, an, the|
|INTJ| interjection(감탄사) |psst, ouch, bravo, hello|
|NOUN| noun(명사) |girl, cat, tree, air, beauty|
|NUM| numeral(숫자) |1, 2017, one, IV, seventy-six|
|PART| particle(소사) |'s, not|
|PRON| pronoun(대명사) |I, you, myself, somebody|
|PROPN| proper noun(고유명사) |Mary, London, NATO, John|
|PUNCT| punctuation(구두점) |., (, ), ?|
|SCONJ| subordinating conjunction(종속 접속사) |if, while, that|
|SYM| symbol(심볼) |$, %, +, :)|
|VERB| verb(동사) | run, runs, running, eat, ate, eating|
|x| other(기타) | asdfasdfasdfjj|
|SPACE| space | |

### POS Tagging하는 방법
기존에 POS Tagging은 모두 손으로 직접 했기 때문에, 통계적 모델을 구축할 만한 방대한 데이터가 존재합니다.   
*Brown corpus*는 굉장히 잘 정리된 corpus 중 하나입니다.   
POS Tagging에는 **Hidden Markov Model**이 사용됩니다.   

Hidden Markov Model은 sequence가 주어졌을 때 사용되며 기존 단어가 주어졌을 때 다음 단어의 POS Tag를 예측할 수 있습니다.   
예를 들어, the 뒤에 나오는 단어는 40%의 확률로 명사이고, 35% 확률로 형용사이고, 25% 확률로 숫자입니다.   

통계적인 모델과는 다르게, rule-based POS tagger도 존재합니다.   
당연하게도, 이 방법은 통계적인 방법을 전혀 쓰지 않는 것은 아니고, 더 적게 의존할 뿐입니다.   

### POS Tagging하는 이유
POS Tagging하는 것이 유용해 보이기는 하지만, 이 정보로 무엇을 할 수 있을까요?   
POS tag는 역사적으로 다양한 이유와 용도로 사용되어 왔습니다.   
Speech-to-Text 변환과 언어간 변환에서 동음이의어의 모호성을 제거할 수 있습니다.   
예를 들어, *I am going to fish a fish*를 다른 언어로 번역할 때, 앞의 fish는 동사이고 뒤의 fish는 명사임을 알아야 정확한 번역이 가능합니다.   

또한, POS Tagging은 **Dependency Parsing**에 사용됩니다.   
Dependency Parsing은 단어와 단어 사이의 의존도, 또는 관계를 인식하는 것입니다.   
이는 나중에 더 자세히 설명하도록 하겠습니다.   