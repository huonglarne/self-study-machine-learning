import os

import numpy as np

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet

from matplotlib import pyplot as plt

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

def read_date(path):
    date = open(path).read().splitlines()
    date = [x.split(" : ") for x in date]
    date = [row[1] for row in date]
    return date

date = read_date("date0.txt")

def read_article(path, filename):
    return open(os.path.join(path, str(filename)), 'r', encoding='cp437').read().split(",")

def read_article_length(path):
    names = os.listdir(path)
    sorted_names = [each for each in names]
    sorted_names.sort()
    articles = [read_article(path, filename) for filename in sorted_names]
    length = [len(article) for article in articles]
    return articles, length


articles, length = read_article_length("articles27.08-27.09")

id2word = corpora.Dictionary(articles)
corpus = [id2word.doc2bow(text) for text in articles]

num_topics = 15

os.environ['MALLET_HOME'] = r"mallet-2.0.8/mallet-2.0.8"
mallet_path = r"mallet-2.0.8/mallet-2.0.8/bin/mallet"
lda_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)

# topics = lda_model.show_topics(formatted=False)
topics = lda_model.print_topics()
print(type(lda_model.print_topics()))

with open('topics0.txt', 'w') as f:
    for item in topics:
        f.write(str(item))
        f.write("\n")

# Compute Coherence Score
# coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=articles, dictionary=id2word, coherence='c_v')
# coherence_ldamallet = coherence_model_ldamallet.get_coherence()
# pprint('\nCoherence Score: ', coherence_ldamallet)

def count_topics(lda_model=lda_model, corpus=corpus):
    print(len(lda_model[corpus]), num_topics)
    count = np.zeros((len(lda_model[corpus]), num_topics))
    for i, row in enumerate(lda_model[corpus]):
        # print(row)
        for (topic_num, prop_topic) in row:
            count[i, topic_num] = round(prop_topic, 4)
    return count

topic_frequency = count_topics(lda_model=lda_model, corpus=corpus)

day_count = []
temp = 0
while temp<len(topic_frequency):
    start = temp
    while temp<len(topic_frequency) and date[temp] == date[start]:
        temp += 1
    sum_day = [sum(topic_frequency[start:temp-1, topic]) for topic in range(num_topics)]
    day_count.append(sum_day)

day_count = np.asarray(day_count)
day_count = np.flip(day_count)

figure = plt.figure()

for i in range(num_topics):
    plt.subplot(5, 3, i+1)
    plt.plot(day_count[:,i], label = i)

plt.legend()
plt.show()
plt.close()



