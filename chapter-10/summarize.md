## 텍스트 요약
---

텍스트 분석에서, 긴 길이의 텍스트를 요약하는 것은 굉장히 중요합니다.   
이번 단원에서는 우리만의 pipeline을 구축하는 것보다는 Gensim의 API를 사용하는데 초점을 두겠습니다.   

기억해야 할 것은 Gensim은 요약할 때 자신만의 문장을 만들지 않고 주어진 text에서 key sentences를 뽑아냅니다.   
이 summarizer는 TextRank algorithm에 기반해서 만들어졌습니다.   

```
from gensim.summarization import summarize

text = "Eleven-year-old Harry Potter has been living an ordinary life, constantly abused by his surly and cold uncle and aunt, Vernon and Petunia Dursley, and bullied by their spoiled son, Dudley. Hagrid explains Harry's hidden past as the wizard son of James and Lily Potter, who are a wizard and witch, respectively, and how they were murdered by the most evil and powerful dark wizard in history, Lord Voldemort, which resulted in the one-year-old Harry being sent to live with his aunt and uncle. There, Harry also makes an enemy of yet another first-year, Draco Malfoy, who prejudices against Hermione due to her being the daughter of Muggles, a term used by wizards and witches, which describes ordinary humans with no magical ability. He winds up in Gryffindor instead with Ron and Hermione while Draco is sorted into Slytherin, like his whole family before him. As classes begin at Hogwarts, Harry discovers his innate talent for flying on broomsticks despite no prior experience and is recruited into his House's Quidditch (a competitive wizards' sport, played in the air) team as a Seeker, which is said to be the most difficult role. When the school's headmaster Albus Dumbledore is lured from Hogwarts under false pretenses, Harry, Hermione, and Ron fear that the theft is imminent and descend through the trapdoor themselves. The eventful school year ends at the final feast, during which Gryffindor wins the House Cup. Harry returns to Privet Drive for the summer, neglecting to tell them that the use of spells is forbidden by under-aged wizards and witches and thus anticipating some fun and peace over the holidays."

print(summarize(text))
```

'해리 포터의 현자의 돌'의 일부분을 발췌했습니다.   
요약을 위해 간단히 summarize 함수만 불러오면 됩니다.   

결과는 다음과 같습니다.   

```
Eleven-year-old Harry Potter has been living an ordinary life, constantly abused by his surly and cold uncle and aunt, Vernon and Petunia Dursley, and bullied by their spoiled son, Dudley.
```

단어의 개수를 제한해서 실행할 수도 있습니다.   

```
print(summarize(text, word_count=50))
```

```
Eleven-year-old Harry Potter has been living an ordinary life, constantly abused by his surly and cold uncle and aunt, Vernon and Petunia Dursley, and bullied by their spoiled son, Dudley.
There, Harry also makes an enemy of yet another first-year, Draco Malfoy, who prejudices against Hermione due to her being the daughter of Muggles, a term used by wizards and witches, which describes ordinary humans with no magical ability.
```

키워드를 뽑아내 보겠습니다.   

```
from gensim.summarization import keywords

print(keywords(text))
```

결과는 다음과 같습니다.   

```
harry
wizard
wizards
year
ordinary
son
lord
gryffindor
albus
life constantly
school
draco
dark
sport
cup
```

Gensim은 Montemurro와 Zanette의 entropy를 이용해 keyword를 뽑아내는 다른 방법도 갖고 있습니다.   
이 방식은 더 큰 corpus에서 더 잘 작동합니다.   

```
from gensim.summarization import mz_keywords

print(mz_keywords(text, scores=True, weighted=False, threshold=1.0))
```