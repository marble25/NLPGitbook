## POS Tagger 학습시키기
---

### Pos Tagger 학습
spaCy의 POS tag 예측은 통계 모델을 기반으로 합니다.   
통계 모델이라는 말은, 우리가 더 나은 예측을 하기 위해 모델을 학습시킬 수 있다는 말입니다.   
현재 spaCy 모델은 tagging에 있어서 97%의 정확도를 자랑하고 있습니다.   

training하기 전에 training이 어떻게 이루어지는지 알고 갑시다.   
우리가 정리한 데이터를 모델에게 제공하고, 모델은 에러를 최소화하는 방향으로 가중치(weights)를 업데이트합니다.   

이렇게 모든 데이터에 대해 training이 끝나면, testing data를 통한 테스트 과정이 필요합니다.   
얼마나 데이터가 잘 학습되었는지 확인하는 과정입니다.   
이것 역시 정리된 데이터를 통해 확인하고, 실제 태그와 모델이 예측한 태그를 비교해서 얼마나 맞추었는지를 확인합니다.   

데이터를 얻는 것은 굉장히 힘든 일이고, 큰 프로젝트에서는 이 것이 병목이 될 수 있습니다.   
이를 위해 prodigy tool이나 GoldParse tool이 있습니다.   
하지만 우리는 이번에는 그에 대해 다루지 않을 것입니다.   

간단한 training의 예시는 다음과 같습니다.   

```
import spacy
import random

TRAIN_DATA = [
    ("Facebook has been accused for leaking personal data of users.", {'entities': [(0, 8, 'ORG')]}),
    ("Tinder uses sophisticated algorithms to find the perfect match.", {'entities': [(0, 6, 'ORG')]})
]

nlp = spacy.blank('en')
optimizer = nlp.begin_training()
for i in range(20):
    random.shuffle(TRAIN_DATA)
    for text, annotations in TRAIN_DATA:
        nlp.update([text], [annotations], sgd=optimizer)

nlp.to_disk('model')
```

그저 sentence와 label(entities, heads, deps, tags, cats 등)을 함께 제공하기만 하면 됩니다.   
이 예제에서는 Facebook과 Tinder가 ORG entities로 주어졌습니다.   

POS tagger 역시 그렇게 다르지 않습니다.   
spaCy github에 있는 코드를 이용해서 어떻게 하는지 알아보겠습니다.   

```
import spacy
import random
from pathlib import Path
import plac

TAG_MAP = {
    'N': {'pos': 'NOUN'},
    'V': {'pos': 'VERB'},
    'J': {'pos': 'ADJ'}
}

TRAIN_DATA = [
    ("I like green eggs", {'tags': ['N', 'V', 'J', 'N']}),
    ("Eat blue ham", {'tags': ['V', 'J', 'N']})
]
```

우리들이 사용할 TAG_MAP을 새로 정의해서 TRAIN_DATA에 그 태그대로 넣었습니다.   

```
@plac.annotations(
    lang=("ISO Code of language to use", "option", "l", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iretations", "option", "n", int))
```

몇 가지 덧붙일 것들을 넣었습니다.   

```
def main(lang='en', output_dir=None, n_iter=25):
    nlp = spacy.blank(lang)
    tagger = nlp.create_pipe('tagger')

    for tag, values in TAG_MAP.items():
        tagger.add_label(tag, values)
    nlp.add_pipe(tagger)
```

실제 데이터 안에 우리가 정의한 태그들을 spaCy 타입으로 변환하는 pipeline을 넣었습니다.   

```
    optimizer = nlp.begin_training()
    for i in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}

        for text, annotations in TRAIN_DATA:
            nlp.update([text], [annotations], sgd=optimizer, losses=losses)
        print(losses)
```

TRAIN_DATA를 이용해서 학습을 진행하고, 에러(loss)를 loss에 담아 출력합니다.   

```
    test_text = "I like blue eggs"
    doc = nlp(test_text)
    print('Tags', [(t.text, t.tag_, t.pos_) for t in doc])

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

        nlp.to_disk(output_dir)
        print('Saved model to', output_dir)

        print('Loading from', output_dir)
        nlp2 = spacy.load(output_dir)
        doc = nlp2(test_text)
        print('Tags', [(t.text, t.tag_, t.pos_) for t in doc])
```

test_text를 만들어서 예측한 결과를 출력해 봅니다.   
output_dir이 주어져 있다면, 그 폴더에 모델을 저장하고 불러오는 것도 해 봅니다.   

이 것이 우리가 학습한 custom POS-tagger입니다.   
당연히, 데이터가 굉장히 작기 때문에 좋은 POS tagger는 아니지만, 이런 식으로 학습을 할 수 있다는 것을 보여줍니다.   