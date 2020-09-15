## 텍스트를 위한 딥러닝
---

우리는 word embeddings를 사용하면서 neural network의 강력함을 이미 경험했습니다.   
이 외에도 다양한 텍스트 분석 작업에 neural network를 사용할 수 있습니다.   

유명한 예시 중 하나는 Language Translation으로 특히 Google의 Neural Translation 모델이 유명합니다.   
구글에서는 **zero-shot translation** 방식으로 번역하는데, 이는 언어를 번역할 때에 중간 언어를 거쳐서 번역한다는 이야기입니다.   
예를 들자면, 말레이시아어에서 아랍어로 번역할 때에 영어로 번역한 후, 영어를 아랍어로 번역하는 방식입니다.   
문장을 쪼개거나, rule-base 기반의 번역을 하기보다, 이제는 neural network를 이용해서 번역을 하게 됩니다.   
하지만 기계 번역에 있어서의 발전에도 불구하고, 이 것은 여전히 어려운 작업입니다.   

Word Embeddings는 Text에서 딥 러닝이 쓰이는 또다른 예시 중 하나입니다.   
Word Vectors와 Document Vectors가 얼마나 많은 NLP task에서 사용되는지를 고려하면, 텍스트 처리에 딥 러닝이 굉장히 중요함을 알 수 있습니다.   
또한, clustering과 classification의 기술들에도 neural network가 자주 쓰입니다.   
Sentiment Analysis(감정 분석)에는 CNN이나 RNN과 같은 더 복잡한 neural networks가 쓰이기는 하지만, 기본적으로 간단한 neural network도 사용될 수 있습니다.   

POS tagger와 NER tagger를 학습시킬 때에도 내부적으로 neural network가 사용되었습니다.   
우리가 spaCy에서 훈련된 POS tagger만 쓴다 하더라도, 딥 러닝 요소를 사용한다고 말할 수 있는 셈입니다.   

우리는 단어나 문자가 앞에 올 확률을 이용하면 sequence generator를 만들어 낼 수 있고, 이제 neural network 모델은 generative model이 됩니다.   
다음 장에서는 이 흥미로운 주제에 대해서 더 자세히 알아볼 것입니다.   