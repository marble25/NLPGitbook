## Latent Semantic Analysis
---

### Latent Semantic Analysis란?
LSA는 DTM을 차원 축소 하여 축소 차원에서 근접 단어들을 토픽으로 묶는 방식입니다.   
이 역시 간단히 gensim에서 생성할 수 있습니다.   

```
lsimodel = LsiModel(corpus=corpus, num_topics=10, id2word=dictionary)
print(lsimodel.show_topics(num_topics=5))
```

결과는 다음과 같습니다.   

```
[(0, '-0.229*"israeli" + -0.212*"Arafat" + -0.194*"palestinian" + -0.176*"force" + -0.158*"kill" + -0.157*"official" + -0.150*"attack" + -0.142*"people" + -0.119*"day" + -0.115*"Israel"'), (1, '0.307*"israeli" + 0.306*"Arafat" + 0.273*"palestinian" + 0.165*"Sharon" + -0.161*"Afghanistan" + -0.158*"Australia" + 0.155*"Israel" + 0.127*"Hamas" + 0.124*"West_Bank" + -0.114*"day"'), (2, '-0.262*"Afghanistan" + -0.221*"force" + 0.187*"fire" + -0.185*"Al_Qaeda" + -0.175*"bin_Laden" + -0.153*"Pakistan" + 0.145*"Sydney" + -0.133*"fighter" + -0.131*"Tora_Bora" + -0.128*"Taliban"'), (3, '0.382*"fire" + 0.269*"area" + -0.208*"Australia" + 0.199*"Sydney" + 0.177*"firefighter" + 0.158*"north" + 0.149*"wind" + 0.134*"Wales" + 0.134*"New_South" + 0.127*"south"'), (4, '0.274*"company" + 0.205*"Qantas" + 0.183*"union" + -0.159*"test" + 0.147*"worker" + -0.137*"win" + -0.135*"match" + -0.135*"South_Africa" + -0.127*"wicket" + 0.121*"Australian"')]
```

LDA와 비슷한 결과를 가져오는 것을 볼 수 있습니다.   
일단은 숫자에 마이너스 기호를 붙인 것을 무시해도 괜찮습니다.   
LSI를 진행하는 도중에 SVD(Singular-value Decomposition)을 사용하면서 생긴 것이기 때문입니다.   
이에 대한 자세한 내용은 *Indexing by Latent Semantic Analysis* 논문과 *Probabilistic latent semantic indexing*에서 찾아볼 수 있습니다.   