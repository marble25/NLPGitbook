## 텍스트 생성
---

이번 단원에서는 딥 러닝 과제를 해결하기 위해 keras를 사용할 것입니다.   
keras 설치는 다음과 같이 할 수 있습니다.   

```
pip install --upgrade pip
pip install keras
pip install tensorflow
```

이전에 사용했던 'HP1.txt'를 사용해서 진행하겠습니다.   

```
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import numpy as np

filename = 'HP1.txt'
raw_text = open(filename).read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
chars_to_int = dict((c, i) for i, c in enumerate(chars))
ix_to_char = dict((i, c) for i, c in enumerate(chars))
vocab_size = len(chars)
```

HP1을 읽어서 `chars`에는 이 텍스트에 나오는 모든 문자를 저장하고, 인덱스와 문자 사이의 매핑 테이블을 만들어 주었습니다.   

```
seq_length = 100
list_X = []
list_Y = []
for i in range(0, len(raw_text) - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    list_X.append([chars_to_int[char] for char in seq_in])
    list_Y.append([chars_to_int[seq_out]])

n_patterns = len(list_X)

X = np.reshape(list_X, (n_patterns, seq_length, 1))
Y = np_utils.to_categorical(list_Y)
```

100개의 문자로 sliding window를 만들어서 문맥을 제공했습니다.   
그래서 100개의 문자를 제공하면 그 다음 문자를 맞출 수 있도록 list_X와 list_Y에 문자의 index 형태로 바꾸어 저장했습니다.   

```
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

이 경우에는 LSTM을 하나의 레이어로 제공하고, 0.2의 Dropout, SoftMax activation, 그리고 ADAM optimizer를 적용했습니다.   
Dropout은 하나의 데이터셋에만 overfitting되지 않도록 노드의 수를 줄여주는 것입니다.   
Activation은 네트워크에서 어떤 뉴런을 활성시킬지를 결정해주고, optimizer는 에러를 줄여주는 역할을 통해 실질적인 학습을 진행합니다.   

```
filepath='weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(X, Y, epochs=20, batch_size=128, callbacks=callbacks_list)
```

모델을 완성했으면 실제로 학습을 진행해 보겠습니다.   
학습은 꽤 오랜 시간이 지난 후에 완료가 됩니다.   
`fit` 함수에서 실제로 파라미터들을 가지고 학습을 진행하게 됩니다.   

```
start = np.random.randint(0, len(X) - 1)
pattern = np.ravel(X[start]).tolist()

output = []
for i in range(250):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_size)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = index
    output.append(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print('"{}"'.format(''.join([ix_to_char[value] for value in output])))
```

처음에 랜덤으로 하나의 캐릭터로 시작해서 argmax를 이용해서 그 다음에 올 가장 높은 character를 골라서 붙여나가는 형식입니다.   

위의 LSTM 예시에서는 그렇게 많은 레이어를 쌓지 않았습니다.   
하지만, 더 많은 레이어를 쌓고 파라미터를 조정한다면, 훨씬 좋은 결과를 가져올 수 있습니다.   
