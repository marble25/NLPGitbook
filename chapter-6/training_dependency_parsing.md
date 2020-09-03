## Dependency Parsing 학습시키기
---

### Dependency Parser 학습(기초)
이전 예제와 동일하게, [train_parser.py](https://github.com/explosion/spaCy/blob/master/examples/training/train_parser.py)를 실습해 보면서 Dependency Parser를 학습해 보겠습니다.   

```
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# training data
TRAIN_DATA = [
    (
        "They trade mortgage-backed securities.",
        {
            "heads": [1, 1, 4, 4, 5, 1, 1],
            "deps": ["nsubj", "ROOT", "compound", "punct", "nmod", "dobj", "punct"],
        },
    ),
    (
        "I like London and Berlin.",
        {
            "heads": [1, 1, 1, 2, 2, 1],
            "deps": ["nsubj", "ROOT", "dobj", "cc", "conj", "punct"],
        },
    ),
]
```

이전 예제들과 동일하게, Train data에는 해당 문장을 넣고, 그에 해당하는 head와 dependency label을 함께 넣어주었습니다.   
heads의 경우에는 word로 구분했을 때 head가 되는 index를 의미합니다.   
예륻 들어 They의 경우 heads가 1이므로 1번째 있는 trade가 head가 되고, mortgage의 경우 heads가 4이므로 4번째 있는 backed가 head가 됩니다.   
deps의 경우에는 각자에 해당하는 dependency label을 넣어주면 됩니다.   

```
@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=15):
    """Load the model, set up the pipeline and train the parser."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # add the parser to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "parser" not in nlp.pipe_names:
        parser = nlp.create_pipe("parser")
        nlp.add_pipe(parser, first=True)
    # otherwise, get it, so we can add labels to it
    else:
        parser = nlp.get_pipe("parser")
```

이 단계는 이전 예제와 유사하게 blank model을 불러와서 학습을 하는 과정입니다.   

```
    # add labels to the parser
    for _, annotations in TRAIN_DATA:
        for dep in annotations.get("deps", []):
            parser.add_label(dep)

    # get names of other pipes to disable them during training
    pipe_exceptions = ["parser", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train parser
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, losses=losses)
            print("Losses", losses)
```

minibatch를 이용해서 4.0에서 시작해 32.0이 될 때까지 1.001만큼 곱해주면서 학습한다는 의미입니다.   
nlp process는 간단하게 update를 이용해서 진행됩니다.   

```
    # test the trained model
    test_text = "I like securities."
    doc = nlp(test_text)
    print("Dependencies", [(t.text, t.dep_, t.head.text) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc = nlp2(test_text)
        print("Dependencies", [(t.text, t.dep_, t.head.text) for t in doc])


if __name__ == "__main__":
    plac.call(main)

    # expected result:
    # [
    #   ('I', 'nsubj', 'like'),
    #   ('like', 'ROOT', 'like'),
    #   ('securities', 'dobj', 'like'),
    #   ('.', 'punct', 'like')
    # ]
```

이와 같이 실행을 마치면 결과는 다음과 같습니다.   

```
Dependencies [('I', 'nsubj', 'like'), ('like', 'ROOT', 'like'), ('securities', 'dobj', 'like'), ('.', 'conj', 'securities')]
```

위의 예시는 굉장히 기초적인 parsing을 진행한 것입니다.   

### Dependency Parser 학습(custom)
우리가 정의한 label에 따른 Dependency Parser를 학습해 보겠습니다.   
파일은 [train_intent_parser.py](https://github.com/explosion/spaCy/blob/master/examples/training/train_intent_parser.py)를 참조했습니다.   

```
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# training data: texts, heads and dependency labels
# for no relation, we simply chose an arbitrary dependency label, e.g. '-'
TRAIN_DATA = [
    (
        "find a cafe with great wifi",
        {
            "heads": [0, 2, 0, 5, 5, 2],  # index of token head
            "deps": ["ROOT", "-", "PLACE", "-", "QUALITY", "ATTRIBUTE"],
        },
    ),
    (
        "find a hotel near the beach",
        {
            "heads": [0, 2, 0, 5, 5, 2],
            "deps": ["ROOT", "-", "PLACE", "QUALITY", "-", "ATTRIBUTE"],
        },
    ),
    (
        "find me the closest gym that's open late",
        {
            "heads": [0, 0, 4, 4, 0, 6, 4, 6, 6],
            "deps": [
                "ROOT",
                "-",
                "-",
                "QUALITY",
                "PLACE",
                "-",
                "-",
                "ATTRIBUTE",
                "TIME",
            ],
        },
    ),
    (
        "show me the cheapest store that sells flowers",
        {
            "heads": [0, 0, 4, 4, 0, 4, 4, 4],  # attach "flowers" to store!
            "deps": ["ROOT", "-", "-", "QUALITY", "PLACE", "-", "-", "PRODUCT"],
        },
    ),
    (
        "find a nice restaurant in london",
        {
            "heads": [0, 3, 3, 0, 3, 3],
            "deps": ["ROOT", "-", "QUALITY", "PLACE", "-", "LOCATION"],
        },
    ),
    (
        "show me the coolest hostel in berlin",
        {
            "heads": [0, 0, 4, 4, 0, 4, 4],
            "deps": ["ROOT", "-", "-", "QUALITY", "PLACE", "-", "LOCATION"],
        },
    ),
    (
        "find a good italian restaurant near work",
        {
            "heads": [0, 4, 4, 4, 0, 4, 5],
            "deps": [
                "ROOT",
                "-",
                "QUALITY",
                "ATTRIBUTE",
                "PLACE",from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# training data: texts, heads and dependency labels
# for no relation, we simply chose an arbitrary dependency label, e.g. '-'
TRAIN_DATA = [
    (
        "find a cafe with great wifi",
        {
            "heads": [0, 2, 0, 5, 5, 2],  # index of token head
            "deps": ["ROOT", "-", "PLACE", "-", "QUALITY", "ATTRIBUTE"],
        },
    ),
    (
        "find a hotel near the beach",
        {
            "heads": [0, 2, 0, 5, 5, 2],
            "deps": ["ROOT", "-", "PLACE", "QUALITY", "-", "ATTRIBUTE"],
        },
    ),
    (
        "find me the closest gym that's open late",
        {
            "heads": [0, 0, 4, 4, 0, 6, 4, 6, 6],
            "deps": [
                "ROOT",
                "-",
                "-",
                "QUALITY",
                "PLACE",
                "-",
                "-",
                "ATTRIBUTE",
                "TIME",
            ],
        },
    ),
    (
        "show me the cheapest store that sells flowers",
        {
            "heads": [0, 0, 4, 4, 0, 4, 4, 4],  # attach "flowers" to store!
            "deps": ["ROOT", "-", "-", "QUALITY", "PLACE", "-", "-", "PRODUCT"],
        },
    ),
    (
        "find a nice restaurant in london",
        {
            "heads": [0, 3, 3, 0, 3, 3],
            "deps": ["ROOT", "-", "QUALITY", "PLACE", "-", "LOCATION"],
        },
    ),
    (
        "show me the coolest hostel in berlin",
        {
            "heads": [0, 0, 4, 4, 0, 4, 4],
            "deps": ["ROOT", "-", "-", "QUALITY", "PLACE", "-", "LOCATION"],
        },
    ),
    (
        "find a good italian restaurant near work",
        {
            "heads": [0, 4, 4, 4, 0, 4, 5],
            "deps": [
                "ROOT",
                "-",
                "QUALITY",
                "ATTRIBUTE",
                "PLACE",
                "ATTRIBUTE",
                "LOCATION",
            ],
        },
    ),
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=15):
    """Load the model, set up the pipeline and train the parser."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # We'll use the built-in dependency parser class, but we want to create a
    # fresh instance – just in case.
    if "parser" in nlp.pipe_names:
        nlp.remove_pipe("parser")
    parser = nlp.create_pipe("parser")
    nlp.add_pipe(parser, first=True)

    for text, annotations in TRAIN_DATA:
        for dep in annotations.get("deps", []):
            parser.add_label(dep)

    pipe_exceptions = ["parser", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train parser
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, losses=losses)
            print("Losses", losses)

    # test the trained model
    test_model(nlp)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        test_model(nlp2)


def test_model(nlp):
    texts = [
        "find a hotel with good wifi",
        "find me the cheapest gym near work",
        "show me the best hotel in berlin",
    ]
    docs = nlp.pipe(texts)
    for doc in docs:
        print(doc.text)
        print([(t.text, t.dep_, t.head.text) for t in doc if t.dep_ != "-"])


if __name__ == "__main__":
    plac.call(main)

    # Expected output:
    # find a hotel with good wifi
    # [
    #   ('find', 'ROOT', 'find'),
    #   ('hotel', 'PLACE', 'find'),
    #   ('good', 'QUALITY', 'wifi'),
    #   ('wifi', 'ATTRIBUTE', 'hotel')
    # ]
    # find me the cheapest gym near work
    # [
    #   ('find', 'ROOT', 'find'),
    #   ('cheapest', 'QUALITY', 'gym'),
    #   ('gym', 'PLACE', 'find'),
    #   ('near', 'ATTRIBUTE', 'gym'),
    #   ('work', 'LOCATION', 'near')
    # ]
    # show me the best hotel in berlin
    # [
    #   ('show', 'ROOT', 'show'),
    #   ('best', 'QUALITY', 'hotel'),
    #   ('hotel', 'PLACE', 'show'),
    #   ('berlin', 'LOCATION', 'hotel')
    # ]

                "ATTRIBUTE",
                "LOCATION",
            ],
        },
    ),
]
```

이 예제에서는 ROOT, PLACE, QUALITY, ATTRIBUTE, LOCATION 등의 dependency label을 설정했고, 그 외의 것에 대해서는 - label을 붙였습니다.   
이 dependency label에 따라 여러 개의 TRAIN_DATA를 준비했습니다.   

```
@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=15):
    """Load the model, set up the pipeline and train the parser."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # We'll use the built-in dependency parser class, but we want to create a
    # fresh instance – just in case.
    if "parser" in nlp.pipe_names:
        nlp.remove_pipe("parser")
    parser = nlp.create_pipe("parser")
    nlp.add_pipe(parser, first=True)

    for text, annotations in TRAIN_DATA:
        for dep in annotations.get("deps", []):
            parser.add_label(dep)

    pipe_exceptions = ["parser", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train parser
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, losses=losses)
            print("Losses", losses)
```

앞서 제시했던 기초적인 parser과 거의 유사하게 training을 진행했습니다.   


```
    # test the trained model
    test_model(nlp)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        test_model(nlp2)


def test_model(nlp):
    texts = [
        "find a hotel with good wifi",
        "find me the cheapest gym near work",
        "show me the best hotel in berlin",
    ]
    docs = nlp.pipe(texts)
    for doc in docs:
        print(doc.text)
        print([(t.text, t.dep_, t.head.text) for t in doc if t.dep_ != "-"])


if __name__ == "__main__":
    plac.call(main)

    # Expected output:
    # find a hotel with good wifi
    # [
    #   ('find', 'ROOT', 'find'),
    #   ('hotel', 'PLACE', 'find'),
    #   ('good', 'QUALITY', 'wifi'),
    #   ('wifi', 'ATTRIBUTE', 'hotel')
    # ]
    # find me the cheapest gym near work
    # [
    #   ('find', 'ROOT', 'find'),
    #   ('cheapest', 'QUALITY', 'gym'),
    #   ('gym', 'PLACE', 'find'),
    #   ('near', 'ATTRIBUTE', 'gym'),
    #   ('work', 'LOCATION', 'near')
    # ]
    # show me the best hotel in berlin
    # [
    #   ('show', 'ROOT', 'show'),
    #   ('best', 'QUALITY', 'hotel'),
    #   ('hotel', 'PLACE', 'show'),
    #   ('berlin', 'LOCATION', 'hotel')
    # ]
```

Training을 성공적으로 마치고 test를 진행하면 다음과 같이 output이 나옵니다.   

```
find a hotel with good wifi
[('find', 'ROOT', 'find'), ('hotel', 'PLACE', 'find'), ('good', 'QUALITY', 'wifi'), ('wifi', 'ATTRIBUTE', 'hotel')]
find me the cheapest gym near work
[('find', 'ROOT', 'find'), ('cheapest', 'QUALITY', 'gym'), ('gym', 'PLACE', 'find'), ('work', 'LOCATION', 'near')]
show me the best hotel in berlin
[('show', 'ROOT', 'show'), ('best', 'QUALITY', 'hotel'), ('hotel', 'PLACE', 'show'), ('berlin', 'LOCATION', 'hotel')]
```

예상한 것과 같이 잘 나온 것을 알 수 있습니다.   

이 예제를 통해 spaCy가 custom model을 만들 때 그 진가를 발휘함을 알 수 있습니다.   