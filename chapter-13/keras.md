## Keras와 딥러닝
---

### RNN을 이용한 Classification

지난 장에서는 Keras를 이용해서 텍스트를 생성하는 딥 러닝 모델을 구축해서 직접 학습하고, 테스트까지 해 보았습니다.   
이번에는 Keras를 이용해서 classification 과제를 수행해 보겠습니다.   

```
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_features = 20000
maxlen = 80
batch_size = 32
print('Loading data...')

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape: ', x_train.shape)
print('x_test shape: ', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train, batch_size=batch_size, epochs=15, validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Test score: ', score)
print('Test accuracy: ', acc)
```

이번 코드의 데이터는 Keras에 이미 내장되어 있는 imdb 데이터로 진행했습니다.   
imdb 데이터는 이미 preprocessing이 진행되어 있기 때문에 따로 preprocessing을 할 필요는 없습니다.   
imdb 데이터는 라벨로 분류가 되어 있어서 편하게 가져다 쓰기만 하면 됩니다.   

이번 예제에서 네트워크는 Embedding Layer와 LSTM Layer, Sigmoid Layer로 구성되어 있습니다.   
Embedding Layer는 20000개의 단어로 되어 있는 입력을 128개의 차원으로 줄일 것입니다.   
LSTM Layer는 실제로 문맥을 가지고 신경망을 구성하게 됩니다.   
Sigmoid Layer의 경우 LSTM Layer의 출력을 가지고 실제로 분류를 하는 작업을 거치게 됩니다.   

Model을 adam optimizer를 사용해서 구성했고, 학습 후에 실제로 테스트를 진행해 보았습니다.   

코드를 실행하면 꽤 긴 시간의 학습 후에 결과를 얻을 수 있습니다.   

```
Test score:  1.087781548500061
Test accuracy:  0.8176800012588501
```

결과는 다음과 같이 나옵니다.   
정확도가 0.81 정도면 그렇게 나쁘지는 않지만 향상될 여지가 많은 것 같습니다.   

### CNN을 이용한 Classification

CNN을 이용해서 진행하는 예시도 있습니다.   

```
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb

kernel_size = 5
filters = 64
pool_size = 4

max_features = 20000
maxlen = 100
embedding_size = 128

lstm_output_size = 70

batch_size = 30
epochs = 10
print('Loading data...')

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape: ', x_train.shape)
print('x_test shape: ', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Test score: ', score)
print('Test accuracy: ', acc)
```

Parameter를 설정하는 부분이 조금 더 길고, 모델이 조금 더 복잡한 것 외에는 비슷합니다.   
모델을 살펴보면, Embedding으로 차원을 축소하고, Dropout으로 Overfitting을 방지합니다.   
그 후, Conv1D 레이어를 통해 1차원 상에서 5개 단위로 잘라 보면서 특징을 추출하고, MaxPooling1D를 통해 그 layer들의 최대치를 종합합니다.   
LSTM으로 문맥 이해를 한 후, Dense Layer를 거쳐 Activation Layer로 가게 됩니다.   

결과는 다음과 같습니다.   

```
Test score:  0.35533609986305237
Test accuracy:  0.8527200222015381
```

단순히 RNN만 적용한 것보다 다양한 Layer로 복잡하게 구성한 CNN Network가 유의미한 정확도 상승을 이뤄냈습니다.   

### Word Embedding 사용하기

이번에는 Word Embedding을 이용해서 더 정확도를 높인 모델을 만들어 보겠습니다.   
여기에 사용되는 glove data는 [glove](http://nlp.stanford.edu/data/glove.6B.zip)에서 다운로드 받을 수 있습니다.   
여기에 사용되는 newsgroup data는 [newsgroup](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html)에서 다운로드 받을 수 있습니다.   


```
from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant


BASE_DIR = ''
GLOVE_DIR = ''
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                with open(fpath, **args) as f:
                    t = f.read()
                    i = t.find('\n\n')  # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                labels.append(label_id)

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))

score, acc = model.evaluate(x_val, y_val, batch_size=128)

print('Test score: ', score)
print('Test accuracy: ', acc)
```

결과는 다음과 같습니다.   

```
Test score:  0.9687448740005493
Test accuracy:  0.7211803197860718
```

아까 전과는 데이터셋도 다르고, 모델 구성도 조금 다르기 때문에 오히려 이전 결과보다 더 낮게 나올 수 있습니다.   

이처럼 Word Embedding 중 하나인 GloVe를 사용해서 유의미한 정확도 상승을 이뤄낼 수 있습니다.   
과정은 간단합니다.   
먼저 저장된 GloVe 데이터를 불러와서 메모리에 적재합니다.   
dataset을 토큰화하는 등의 preprocessing을 진행한 후에 glove data를 이용해서 embedding 층을 만들어냅니다.   
그 후, Conv1D와 MaxPooling을 여러개 쌓아서 모델을 구성합니다.   
마지막으로 모델을 훈련시키면 작업은 끝납니다.   

Word Embedding은 model에 대한 더 많은 context를 포함하고 있고, 단어를 더 잘 설명할 수 있기 때문에, 성능 향상이 일어나는 것은 당연합니다.   

