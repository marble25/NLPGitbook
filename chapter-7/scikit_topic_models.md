## scikit-learn에서 topic models
---

### scikit-learn 사용하기
Gensim이 topic model을 위한 유일한 패키지는 아닙니다.   
scikit-learn 또한 LDA와 Non-negative Matrix Factorization(NMF) 기능을 잘 제공합니다.   

Non-negative matrix factorization(NMF)는 text mining에만 국한된 것은 아닙니다.   
NMF는 하나의 matrix V를 두 개의 matrices W와 H로 쪼개는 작업입니다.   
W와 H는 original matrix V를 나타내는데 사용합니다.   

NMF의 또 다른 특징은 matrix가 음이 아닌 element로 이루어져 있어야 한다는 점입니다.   

다음과 같이 scikit-learn을 사용할 수 있도록 준비를 합시다.   

```
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

# dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
# documents = dataset.data

no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(skl_texts)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(skl_texts)
tf_feature_names = tf_vectorizer.get_feature_names()
```

여기서 tfidf는 앞에서 배운 Term Frequency-Inverse Term Frequency이고, tf는 Term Frequency입니다.   
그 후 다음과 같은 코드를 추가합니다.   

```
no_topics = 10

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)
```

NMF는 tfidf를 이용하고, LDA는 tf를 이용합니다.   
결과는 다음과 같이 나옵니다.   

```
Topic 0:
australia government year australian new people man economy minister india
Topic 1:
palestinian arafat israeli israel hamas gaza attack suicide sharon militant
Topic 2:
bin laden afghanistan qaeda al force bora tora taliban afghan
Topic 3:
qantas union worker industrial maintenance dispute freeze wage action relations
Topic 4:
test south africa match day waugh bowler wicket cricket lee
Topic 5:
south sydney wind firefighter area line yacht wales storm new
Topic 6:
river guide adventure canyone court trip australians interlaken swiss accident
Topic 7:
detainee centre woomera detention facility department damage overnight visa night
Topic 8:
hollingworth dr governor abuse general anglican child school allegation statement
Topic 9:
commission hih royal collapse hearing company union report evidence martin
Topic 0:
reid large flight explosive solomon islands centre election try start
Topic 1:
test union qantas worker day south industrial lee australian match
Topic 2:
commission government report india australia royal party union president federal
Topic 3:
russian authority die resolve negotiate rest january sentence catch probably
Topic 4:
australia wicket day catch south space africa test final station
Topic 5:
palestinian israeli arafat attack suicide hamas arrest government gaza sharon
Topic 6:
afghanistan force laden bin taliban al qaeda afghan government united
Topic 7:
year new people south company australian world sydney australia today
Topic 8:
detainee woomera detention centre facility visa building overnight night destroy
Topic 9:
australia rate economy bank cent year job economic cut australian
```
