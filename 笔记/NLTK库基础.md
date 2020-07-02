# 1.介绍

NLP领域中最常用的一个Python库

开源项目

自带分类、分词等功能

强大的社区支持

语料库，语言的实际使用中真实出现过的语言材料

# 2.安装

## 1.安装nltk

```shell
pip install nltk
```

## 2.安装nltk_data（语料库）

```shell
import nltk
nltk.download()
```

输入上述指令，会出现以下窗口

![这里写图片描述](https://img-blog.csdn.net/20180817174205186)

点击download下载即可，这个东西比较大，我全部解压开来有3.6GB，如果觉得下载太慢，请看第4条。

## 3.测试是否安装成功

```shell
from nltk.book import *
```



## 4.其他方式安装nltk_data

1）从其他方式下载nltk_data：

​	github：https://github.com/nltk/nltk_data

​	别人的网盘分享

2）将下载的压缩包解压到对应文件夹

​	找到对应文件夹的方法：上面2 中的Download Directory

或输入 上面3  的指令，会出现下图

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181104164253427.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ppYWppa2FuZ19qams=,size_16,color_FFFFFF,t_70)

## 5.其他工具的安装

1.结巴分词安装

```python
pip install jieba
```

如果第一个太慢，可以使用从国内镜像安装，下面使用阿里云的镜像

```shell
pip install jieba -i http://mirrors.aliyun.com/pypi/simple
```



# 3.数据准备





# 4.经典的文本数据预处理流程

![img](https://img-blog.csdn.net/20180115105206016?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveWMxMjAzOTY4MzA1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## 1.分词

​			将句子拆分成具有语言语义学上有意义的词

### 中英文分词区别

​	英文中，单词之间是以空格作为自然分界符

​	中文中没有一个形式上的分界符，分词比英文复杂的多

### 中英文分词工具

​	中文：jieba分词

​	英文：nltk自带的分词工具

注意：nltk中还自带了很多种语言的分词工具

### NLTK分词

```python
from nltk.tokenize import word_tokenize

# 原始文本
raw_text = "He goes to school by bike."
#分词
raw_words = word_tokenize(raw_text)
print(raw_words)
```

### jieba分词

支持四种分词模式，最常用的有如下两种

```python
import jieba

raw_text_cn = "我爱青岛大学"

# 全模式分词
raw_words_cn = jieba.cut(raw_text_cn, cut_all=True)
print("全模式：" + "/".join(seg_list_cn)) # 全模式

# 精确模式分词(不指定cut_all默认为精准模式)
raw_words_cn = jieba.cut(raw_text_cn, cut_all=False)
print("全模式：" + "/".join(seg_list_cn))  # 精确模式
type(raw_words_cn)

# 如果希望分词后返回一个list
raw_words_cn = jieba.lcut(raw_text_cn, cut_all=False)
type(raw_words_cn)
```

- 精确模式：试图将句子最精确地切开，适合文本分析；
- 全模式：把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；

其他模式自行了解：https://github.com/fxsjy/jieba

### 特殊字符分词

例如：  -.-  >.<   :)   @xxx

使用正则表达式进行处理

## 2.词形问题(Stem/lemma)

这个名是我自己起的，看情况理解（>.<）

词形问题

look，looked，looking，looks

比如当我搜索play basketball时，Bob is playing basketball 也符合我的要求，但对于计算机来说 ，play和playing是两种完全不同的东西，所以我们需要进行转化

### 两种方式

​	stemming，词干提取

​	词干提取主要是采用“缩减”的方法，将词转换为词干，如将“cats”处理为“cat”，将“effective”处理为“effect”。

​	lemmatization：词形还原

​	词形还原主要采用“转变”的方法，将词转变为其原形，如将“drove”处理为“drive”，将“driving”处理为“drive”

### Stemming（词干提取)

三种方式

```python
#PorterStemmer
from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()

words = [porter_stemmer.stem(word) for word in raw_words]
print(words)
```

```python
#SnowballStemmer
from nltk.stem import SnowballStemmer

snowball_stemmer = SnowballStemmer('english')

words = [snowball_stemmer.stem(word) for word in raw_words]
print(words)
```

```python
#LancasterStemmer
from nltk.stem.lancaster import LancasterStemmer

lancaster_stemmer = LancasterStemmer()

words = [lancaster_stemmer.stem(word) for word in raw_words]
print(words)
```

### Lemmatizer（词形还原）

```python
from nltk.stem import WordNetLemmatizer   # WordNet语料库

wordnet_lemmatizer = WordNetLemmatizer()

words = [wordnet_lemmatizer.lemmatize(word) for word in raw_words]
print(words)
```

## 3.滤除停用词

 	为节省存储空间和提高搜索效率，NLP中会自动过滤掉某些字或词

​	停用词都是人工输入，非自动化生成的，形成停用词表

### 去除停用字

```python
import nltk.corpus
# NLTK自带了多种语言的停用词列表，下面是获取英文停用词
sw = stopwords.words('english')

print("NLTK停用词：", list(sw[:7])
```

```python
# 过滤掉停用字(注意此停用字语料库单词均为小写)
filtered_words = [word for word in words if word.lower() not in sw]

print('过滤后：', filtered_words)
```

### 去除标点

```python
import string
# 过滤掉停用字(注意此停用字语料库单词均为小写)
filtered_words = [word for word in filtered_words if word.lower() not in string.punctuation]

print('去除标点后：', filtered_words)
```

![1577930393682](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1577930393682.png)

## 4.词性标注(POS Tag)

### 作用

**A.标准化与词形还原**: 位置标注是词形还原的基础步骤之一，可以帮助把单词还原为基本形式.（（playing   v演奏，玩   n剧本））

**B.有效移除停用词** : 利用位置标记可以有效地去除停用词。

**C.强化基于单词的特征:** 一个机器学习模型可以从一个词的很多方面提取信息，但如果一个词已经标注了词性，那么它作为特征就能提供更精准的信息。

### 词性

![1577958273855](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1577958273855.png)

https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

### 基本用法

```python
from nltk import pos_tag

tagged_word = pos_tag(filtered_words)

print('标注词性后：', tagged_word)
```

### 更好的词形还原

```python
# playing做名词意思是 比赛，演奏；做动词是play的ing形式
word = wordnet_lemmatizer.lemmatize("playing", pos='n')
print(word)


word = wordnet_lemmatizer.lemmatize("playing", pos='v')
print(word)
```

```python
# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
```



### 利用词性标注去除某些单词

```python
new_words = []
for word in tagged_words:
    if word[1] != 'CD':
        new_words.append(word[0])
        
print("去除 '基数'词后：", new_words)
```

# 5.总结以上程序

```python
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
# 原始文本
raw_text = "he goes to school by bike two times a week."

# 分词
raw_words = word_tokenize(raw_text)

# 词形还原
wordnet_lemmatizer = WordNetLemmatizer()
words = [wordnet_lemmatizer.lemmatize(word) for word in raw_words]


# 过滤停用词
sw = stopwords.words('english')
filtered_words = [word for word in words if word.lower() not in sw]

# 利用词性去除某些单词
new_words = []
for word in tagged_words:
    if word[1] != 'CD':
        new_words.append(word[0])
        
print("去除 '基数'词后：", new_words)
```

# 6.Guternberg语料库的使用

nltk自带了一些语料库，例如：Gutenberg，Brown等，每个语料库有对应的函数，下面介绍guternberg的函数

```python
import nltk

gb = nltk.corpus.gutenberg

# raw表示的是文本中所有的标识符
raw = gb.raw("shakespeare-caesar.txt")

# words得到
words = gb.words("shakespeare-caesar.txt")

# sents得到句子
sents = gb.sents("shakespeare-caesar.txt")
```

# 7.TF/TF-IDF

## 1.TF（词频）

文本单词出现的频率或次数

## 2.词频统计

文本预处理工作

```python
import nltk
import string

gb = nltk.corpus.gutenberg
words = gb.words("shakespeare-caesar.txt")

# 创建一个停用词集合
sw = set(nltk.corpus.stopwords.words('english'))
# 创建一个标点符号集合
punctuation = set(string.punctuation)

# 去除停用词和标点符号
filtered_words = [w.lower() for w in words if w.lower() not in sw and w.lower() not in punctuation]
```



词频统计

```python
# 创建一个FreqDist对象
fd = nltk.FreqDist(filtered_words)

# 输出单词和词频
# 注意：python3中使用dict.keys()返回的不再是list类型了，所以此处和课本有出入
print("Words", list(fd.keys())[:5])
print("Counts", list(fd.values())[:5])

# 输出'd'的词频
print("Counts", fd['d'])

# 输出
print("Max", fd.max())
```

由如下继承树可以看出FreqDist继承自dict类，所以可以对FreqDist做字典能做的处理，例如取keys和取values和 使用dict[key]获取对应的value等

![1577933768091](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1577933768091.png)

双字词频/三字词频

使用nltk.bigrams()/nltk.trigrams()将文本转换为两个单词后传入FreqDist即可

```python
# 双字词词频分析
fd = nltk.FreqDist(nltk.bigrams(filtered_words))
print("Bigrams", list(fd.keys())[:5])
print("Counts", list(fd.values())[:5])
print("Bigram Max", fd.max())
print("Bigram count", fd[('let', 'vs')])
```

## 3.TF-IDF（词频-逆文档频率）

一个更好的度量指标，可以理解为      词频*该词在文档中的权重

![1577895320696](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1577895320696.png)

![1577895340329](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1577895340329.png)

![1577895354417](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1577895354417.png)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ["I come to China to travel",
    "This is a car polupar in China",
    "I love tea and Apple ",
    "The work is to write some papers in science"]


tfidf2 = TfidfVectorizer()
re = tfidf2.fit_transform(corpus)
print(re)

print(tfidf2.fit_transform(corpus).toarray())
print(tfidf2.get_feature_names())

```





# 8.词袋模型

​	词袋模型，即它认为一篇文档是由其中的词构成的一个集合（即袋子），词与词之间没有顺序以及先后的关系。

​	通过统计每个词在文本中出现的次数，我们就可以得到该文本基于词的特征，然后利用词频对文本进行向量表示

例如：

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
corpus = ["I go to school by bike",
          "I go to school on foot",
          "I come to China to travel"]
print(vectorizer.fit_transform(corpus))

print(vectorizer.fit_transform(corpus).toarray())
print(vectorizer.get_feature_names())
```

[[1 1 0 0 0 1 0 1 1 0]
 [0 0 0 0 1 1 1 1 1 0]
 [0 0 1 1 0 0 0 0 2 1]]

['bike', 'by', 'china', 'come', 'foot', 'go', 'on', 'school', 'to', 'travel']

# 9.NLTK在NLP的经典应用

情感分析、文本分类、文本相似度

## 1.情感分析

### 是啥

​		文本情感分析(Sentiment Analysis)是指利用自然语言处理和文本挖掘技术，对带有情感色彩的主观性文本进行分析、处理和抽取的过程

### 实现方式

1)情感分析可以通过建立sentiment dictionary(情感字典)来实现。

例如：like-（1）,good-（2）,bad-（-2）,terrible-（-3） 这代表着like的情感正面程度分值为1，good的情感正面程度为2，bad的情感正面程度为-2，terrible的情感正面程度为-3。

当然，这些情感字典的分值是由一群语言学家共同讨论给出。我们可以看到这种方法类似于**关键词打分机制**。



```python

import nltk

# NLTK进行情感分析
# 建立情感字典
sentiment_dictionary = {}
for line in open('C:\Users\Lenovo\Desktop\AFINN\AFINN-111.txt'):
    word, score = line.split('	')
    sentiment_dictionary[word] = int(score)

# 原始文本数据
sentence_1 = 'i love you!'
sentence_2 = 'i hate you!'

#分词
word_list1 = nltk.word_tokenize(sentence_1)
word_list2 = nltk.word_tokenize(sentence_2)

# 遍历每个句子，把每个词的情感得分相加，不在情感字典中的词分数全部置0
s1_score = sum(sentiment_dictionary.get(word, 0) for word in word_list1)
s2_score = sum(sentiment_dictionary.get(word, 0) for word in word_list2)

print('我是句子' + sentence_1 + '的正面情感得分:', s1_score)
print('我是句子' + sentence_2 + '的正面情感得分:', s2_score)

```

但是这种方法存在以下问题：

**（1）出现网络新词不在字典里怎么办？**

**（2）出现特殊词汇怎么办？**

**（3）更深层的语义怎么理解？**



2)通过自己训练语料来使用机器学习进行预测。

```python
from nltk.classify import NaiveBayesClassifier
from nltk import word_tokenize

# 简单手造的训练集
s1 = 'i am a good boy'
s2 = 'i am a handsome boy'
s3 = 'he is a bad boy'
s4 = 'he is a terrible boy'


# 预处理，对出现的单词记录为True
# 形成字典，键：单词    值：词是否出现过
def preprocess(sentence):
    return {word: True for word in word_tokenize(sentence)}


# 给训练集加标签
training_data = [[preprocess(s1), 'pos'],
                 [preprocess(s2), 'pos'],
                 [preprocess(s3), 'neg'],
                 [preprocess(s4), 'neg']
                 ]

# 进行训练
model = NaiveBayesClassifier.train(training_data)
# 预测语句
new_s1 = 'i am a good girl'
new_s2 = 'she is a terrible girl'

# 进行预测
print('我在预测 ' + new_s1 + ' 结果是：', model.classify(preprocess(new_s1)))
print('我在预测 ' + new_s2 + ' 结果是：', model.classify(preprocess(new_s2)))

```



## 2.文本分类

朴素贝叶斯分类

例子：将word分成两类，一类是停用词和标点符号    另一类是：剩下的

```python
import nltk
import string
import random

# 生成停用词集合和标点符号集合
sw = set(nltk.corpus.stopwords.words('english'))
punctuation = set(string.punctuation)


def word_features(word):
    return {'len': len(word)}

# 判断一个词是否一个停用词（包括停用词和标点符号）
def isStopword(word):
    return word in sw or word in punctuation

# 加载gutenberg
gb = nltk.corpus.gutenberg
words = gb.words("shakespeare-caesar.txt")

# 生成列表，每个元素是一个元组，其形式为（单词，单词是否是停用词）
labeled_words = ([(word.lower(), isStopword(word.lower())) for word in words])

# 乱序
random.seed(42)
random.shuffle(labeled_words)
print(labeled_words[:5])

# feature是一个列表，每个元素是一个元组，其形式为（单词长度，是否为停用词）
featuresets = [(word_features(n), word) for (n, word) in labeled_words]

# 切分训练集和测试集
cutoff = int(.9 * len(featuresets))
train_set, test_set = featuresets[:cutoff],featuresets[cutoff:]

# 进行训练
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("'the' class", classifier.classify(word_features('the')))

# 输出准确度
print("Accuracy", nltk.classify.accuracy(classifier, test_set))
# 输出作用最大的特征
print(classifier.show_most_informative_features(5))
```

理解：A={是否为停用词（或标点）}

​			Bi={该词长度为i}



# 10.课本上程序汇总



9.2滤用停用字、姓名和数字

```python
import nltk

# 输出部分StopWord
sw = set(nltk.corpus.stopwords.words('english'))
print("Stop Words", list(sw)[:7])

# 加载Gutenberg语料库，并输出部分文件的名称
gb = nltk.corpus.gutenberg
print("Gutenberg files", gb.fileids()[-5:])

# 从milton-paradise.txt文件中提取前两句内容
text_sent = gb.sents("milton-paradise.txt")[:2]
print("Unfiltered", text_sent)


for sent in text_sent:
    # 给每个词标上词性标注
    filtered = [w for w in sent if w.lower() not in sw]
    print("Filtered", filtered)
    tagged = nltk.pos_tag(filtered)
    print("Tagged", tagged)
    
    # 去除词中的词性标注为NNP和CD的词
    words = []
    for word in tagged:
        if word[1] !='NNP' and word[1] != 'CD':
            words.append(word[0])
            
    print(words)        
    
```

9.3词袋模型

```python
import nltk
from sklearn.feature_extraction.text import CountVectorizer

# 加载Gutenberg语料库
gb = nltk.corpus.gutenberg
hamlet = gb.raw("shakespeare-hamlet.txt")
macbetch = gb.raw("shakespeare-macbeth.txt")


cv = CountVectorizer(stop_words='english')
# 建立特征向量，输出特征向量
print("Feature vector", cv.fit_transform([hamlet, macbetch]).toarray())
# 输出特征向量前5个维度对应的单词
print("Features", cv.get_feature_names()[:5])
```

9.4词频分析

```python
import nltk
import string

gb = nltk.corpus.gutenberg
words = gb.words("shakespeare-caesar.txt")

# 创建一个停用词集合
sw = set(nltk.corpus.stopwords.words('english'))
# 创建一个标点符号集合
punctuation = set(string.punctuation)

# 去除停用词和标点符号
filtered = [w.lower() for w in words if w.lower() not in sw and w.lower() not in punctuation]

# 创建一个FreqDist对象
fd = nltk.FreqDist(filtered)


# 输出单词和词频
# 注意：python3中使用dict.keys()返回的不在是list类型了，所以此处和课本有出入
print("Words", list(fd.keys())[:5])
print("Counts", list(fd.values())[:5])

# 输出
print("Max", fd.max())

# 输出'd'的词频
print("Counts", fd['d'])

# 双字词词频分析
fd = nltk.FreqDist(nltk.bigrams(filtered))
print("Bigrams", list(fd.keys())[:5])
print("Counts", list(fd.values())[:5])
print("Bigram Max", fd.max())
print("Bigram count", fd[('let', 'vs')])
```

9.5朴素贝叶斯分类

```python
import nltk
import string
import random

# 生成停用词集合和标点符号集合
sw = set(nltk.corpus.stopwords.words('english'))
punctuation = set(string.punctuation)


def word_features(word):
    return {'len': len(word)}

# 判断一个词是否一个停用词（包括停用词和标点符号）
def isStopword(word):
    return word in sw or word in punctuation

# 加载gutenberg
gb = nltk.corpus.gutenberg
words = gb.words("shakespeare-caesar.txt")

labeled_words = ([(word.lower(), isStopword(word.lower())) for word in words])
random.seed(42)
random.shuffle(labeled_words)
print(labeled_words[:5])

featuresets = [(word_features(n), word) for (n, word) in labeled_words]
cutoff = int(.9 * len(featuresets))
train_set, test_set = featuresets[:cutoff],featuresets[cutoff:]

classifier = nltk.NaiveBayesClassifier.train(train_set)
print("'the' class", classifier.classify(word_features('the')))

print("Accuracy", nltk.classify.accuracy(classifier, test_set))
print(classifier.show_most_informative_features(5))
```

9.6情感分析

```python
import random
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy
import string

# cat = movie_reviews.categories()
# fid = movie_reviews.fileids(cat[0])
# words = movie_reviews.words(fid[1])
# print(cat)
# print(fid[:5])
# print(words[:5])

# 对影评文档进行标注
# 构建了一个列表，列表里每个元素都是一个列表 加 一个字符串（neg或者pos）
labeled_docs = [(list(movie_reviews.words(fid)), cat) for cat in movie_reviews.categories() for fid in
                movie_reviews.fileids(cat)]
print(labeled_docs[0][0])
print(labeled_docs[0][1])

# 乱序
random.seed(42)
random.shuffle(labeled_docs)

# 获得影评单词的长度
review_words = movie_reviews.words()
print("# Review Words", len(review_words))

# 生成停用词和标点符号集合
sw = set(stopwords.words('english'))
punctuation = set(string.punctuation)


def isStopWord(word):
    return word in sw or word in punctuation


# 滤除停用词和标点
filtered = [w.lower() for w in review_words if not isStopWord(w.lower())]


# 打印滤除之后的长度
print("# After filter", len(filtered))

# 统计词频
words = FreqDist(filtered)
# 选取词频最高的5%的单词作为特征
N = int(.05 * len(words.keys()))
word_features = list(words.keys())[:N]


# 使用原始单词计数来作为度量指标
def doc_features(doc):
    # 统计doc词频
    doc_words = FreqDist(w for w in doc if not isStopWord(w))
    features = {}
    # features是一个字典，key为count(wonderful)形式，值为数字
    for word in word_features:
        features['count(%s)' % word] = (doc_words.get(word, 0))
    return features


# d是一个集合，c是pos或neg
# featuresets是一个列表，每个元素是一个元组，元组的形式为
featuresets = [(doc_features(d), c) for (d, c) in labeled_docs]

# 设置训练集和测试集
train_set, test_set = featuresets[200:], featuresets[:200]
classifier = NaiveBayesClassifier.train(train_set)

# 输出准确度
print("Accuracy", accuracy(classifier, test_set))
# 输出包含信息量最大的特征
print(classifier.show_most_informative_features())

```

