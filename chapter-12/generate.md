## 텍스트 생성
---

이번 단원에서는 딥 러닝 과제를 해결하기 위해 keras를 사용할 것입니다.   
keras 설치는 다음과 같이 할 수 있습니다.   

```
pip install --upgrade pip
pip install keras
pip install tensorflow
```

설치 완료 후 진행해 보겠습니다.   

```
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import random
import io

path = keras.utils.get_file(
    "nietzsche.txt", origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt"
)
with io.open(path, encoding="utf-8") as f:
    text = f.read().lower()
text = text.replace("\n", " ")  # We remove newlines chars for nicer display
print("Corpus length:", len(text))

chars = sorted(list(set(text)))
print("Total chars:", len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
```

니체 파일을 웹에서 읽어들여서 `chars`에는 이 텍스트에 나오는 모든 문자를 저장하고, 인덱스와 문자 사이의 매핑 테이블을 만들어 주었습니다.   

```
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print("Number of sequences:", len(sentences))

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
```

40개의 문자로 sliding window를 만들어서 문맥을 제공했습니다.   
그래서 100개의 문자를 제공하면 그 다음 문자를 맞출 수 있도록 list_X와 list_Y에 문자의 index 형태로 바꾸어 저장했습니다.   

```
model = keras.Sequential(
    [
        keras.Input(shape=(maxlen, len(chars))),
        layers.LSTM(128),
        layers.Dense(len(chars), activation="softmax"),
    ]
)
optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)
```

이 경우에는 LSTM을 하나의 레이어로 제공하고, SoftMax activation, 그리고 RMSprop optimizer를 적용했습니다.   
Activation은 네트워크에서 어떤 뉴런을 활성시킬지를 결정해주고, optimizer는 에러를 줄여주는 역할을 통해 실질적인 학습을 진행합니다.   

```
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
```

모델에서 예측한 값을 가지고 가능성 높은 다음 인덱스를 반환해주는 함수입니다.   

```
epochs = 40
batch_size = 128

for epoch in range(epochs):
    model.fit(x, y, batch_size=batch_size, epochs=1)
    print()
    print("Generating text after epoch: %d" % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print("...Diversity:", diversity)

        generated = ""
        sentence = text[start_index : start_index + maxlen]
        print('...Generating with seed: "' + sentence + '"')

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            sentence = sentence[1:] + next_char
            generated += next_char

        print("...Generated: ", generated)
        print()
```

실제로 학습을 하고 테스트를 진행하는 과정입니다.   
한 epoch마다 학습 후에 랜덤으로 문자열을 가져와서 그 뒤를 예측해 보는 과정입니다.   

위의 LSTM 예시에서는 그렇게 많은 레이어를 쌓지 않았습니다.   
하지만, 더 많은 레이어를 쌓고 파라미터를 조정한다면, 훨씬 좋은 결과를 가져올 수 있습니다.   
