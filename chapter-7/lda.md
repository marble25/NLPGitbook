## Latent Dirichlet Allocation
---

### Latent Dirichlet Allocation란?
가장 유명한 LDA(Latent Dirichlet Allocation)으로 시작하겠습니다.   

gensim에서는 LDA 모델을 간단하게 생성할 수 있습니다.   

```
ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)
print(ldamodel.show_topics())
```

이 코드를 실행하면 다음과 같이 결과가 나옵니다.   

```
[(0, '0.006*"company" + 0.006*"people" + 0.005*"Australia" + 0.005*"Afghanistan" + 0.005*"force" + 0.004*"day" + 0.003*"Qantas" + 0.003*"Australian" + 0.003*"union" + 0.003*"think"'), (1, '0.005*"day" + 0.004*"tell" + 0.004*"call" + 0.003*"year" + 0.003*"time" + 0.003*"United_States" + 0.003*"Afghanistan" + 0.003*"force" + 0.003*"people" + 0.003*"official"'), (2, '0.007*"israeli" + 0.005*"palestinian" + 0.004*"time" + 0.004*"official" + 0.003*"tell" + 0.003*"company" + 0.003*"force" + 0.003*"group" + 0.003*"Arafat" + 0.003*"people"'), (3, '0.006*"attack" + 0.004*"good" + 0.004*"year" + 0.004*"Arafat" + 0.004*"israeli" + 0.004*"official" + 0.004*"palestinian" + 0.004*"kill" + 0.003*"force" + 0.003*"cent"'), (4, '0.006*"arrest" + 0.005*"people" + 0.005*"Sydney" + 0.005*"area" + 0.004*"Australia" + 0.004*"Hamas" + 0.004*"palestinian" + 0.004*"militant" + 0.004*"new" + 0.004*"work"'), (5, '0.005*"Australia" + 0.004*"Government" + 0.004*"day" + 0.004*"year" + 0.004*"new" + 0.003*"fire" + 0.003*"attack" + 0.003*"australian" + 0.003*"people" + 0.003*"hour"'), (6, '0.006*"israeli" + 0.004*"people" + 0.004*"Arafat" + 0.004*"attack" + 0.004*"official" + 0.003*"force" + 0.003*"India" + 0.003*"Pakistan" + 0.003*"kill" + 0.003*"palestinian"'), (7, '0.009*"Australia" + 0.004*"year" + 0.004*"man" + 0.004*"people" + 0.003*"report" + 0.003*"claim" + 0.003*"come" + 0.003*"arrest" + 0.003*"HIH" + 0.003*"australian"'), (8, '0.004*"fire" + 0.004*"man" + 0.004*"people" + 0.004*"Australia" + 0.003*"police" + 0.003*"day" + 0.003*"tell" + 0.003*"include" + 0.003*"today" + 0.003*"force"'), (9, '0.005*"year" + 0.004*"report" + 0.004*"month" + 0.003*"area" + 0.003*"take" + 0.003*"come" + 0.003*"man" + 0.003*"group" + 0.003*"people" + 0.003*"force"')]
```

결과를 자세히 살펴봅시다.   
tuple의 첫 번째 값은 topic id로 topic을 식별하는 id입니다.   

topic 5를 자세히 살펴봅시다.   

```
(5, '0.005*"Australia" + 0.004*"Government" + 0.004*"day" + 0.004*"year" + 0.004*"new" + 0.003*"fire" + 0.003*"attack" + 0.003*"australian" + 0.003*"people" + 0.003*"hour"')
```

topic ID 5는 `Australia`, `Government`, `day` 등의 단어로 이루어져 있고, 이 단어들은 이 주제에서 가장 높은 확률을 보인 단어들입니다.   
단어에 곱해진 float 값들은 단어가 topic distribution에 나타날 확률입니다.   

이 topic을 통해서 사람들이 화기 공격을 했다는 대략적인 주제를 알 수 있습니다.   
