"""
goal convert all the news texts to a tf-idf model.
Possibly have a function that supports processing a document through a tf-idf model.

"""

import logging
import re
from collections import defaultdict
from datetime import datetime
from pprint import pprint

import pandas as pd

from nltk.corpus import stopwords
from gensim import corpora
from gensim import models

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Article:
    def __init__(self,
                 _id=None,
                 title=None,
                 publication=None,
                 author=None,
                 date=None,
                 year=None,
                 month=None,
                 url=None,
                 content=None
                ):
        self._id = _id
        self.title = title
        self.publication = publication
        self.author = author
        self.date = date
        self.year = year
        self.month = month
        self.url = url
        self.content = content

# want list of Article objects
def get_documents_w_metadata():
    documents = []

    #process documents in file 1
    print("reading file 1")
    articles = pd.read_csv("data/articles1.csv", header=0, delimiter=",", quoting=0)
    documents = documents + extract_articles_w_metadata(articles)

    #process documents in file 2
    print("reading file 2")
    articles = pd.read_csv("data/articles2.csv", header=0, delimiter=",", quoting=0)
    documents = documents + extract_articles_w_metadata(articles)

    #process documents in file 3
    print("reading file 3")
    articles = pd.read_csv("data/articles3.csv", header=0, delimiter=",", quoting=0)
    documents = documents + extract_articles_w_metadata(articles)

    print("read %d documents" % (len(documents)))

    return documents

def extract_articles_w_metadata(articles):
    documents = []

    num_articles = len(articles)

    for i in range(num_articles):
        article = Article()
        article.title = articles['title'][i]
        article.publication = articles['publication'][i]
        article.author = articles['author'][i]
        article.date = process_date(articles['date'][i])
        article.year = articles['year'][i]
        article.month = articles['month'][i]
        article.content = articles['content'][i]

        documents.append(article)

    return documents

def process_date(datestring):
    if isinstance(datestring, float):
        return None
    else:
        for fmt in ('%Y-%m-%d', '%Y/%m/%d'):
            try:
                return datetime.strptime(datestring, fmt)
            except ValueError:
                continue
        raise ValueError('no valid date format found for: \"' + datestring + "\"")


# get all the corpora
def get_documents():
    documents = []
    extraction_type = "content"

    # process documents in file 1
    print("reading file 1")
    articles = pd.read_csv("data/articles1.csv", header=0, delimiter=",", quoting=0)
    for article in articles[extraction_type]:
        documents.append(article)

    # process documents in file 2
    print("reading file 2")
    articles = pd.read_csv("data/articles2.csv", header=0, delimiter=",", quoting=0)
    for article in articles[extraction_type]:
        documents.append(article)

    # process documents in file 3
    print("reading file 3")
    articles = pd.read_csv("data/articles3.csv", header=0, delimiter=",", quoting=0)
    for article in articles[extraction_type]:
        documents.append(article)

    print("read %d documents" % (len(documents)))

    return documents


def article_to_wordlist(raw_article, remove_stopwords=True, remove_numbers=True):
    """ convert a string to a wordlist
    """
    if remove_numbers is True:
        # remove non letters
        article_text = re.sub("[^a-zA-Z]", " ", raw_article)
    else:
        # remove non letters non numbers
        article_text = re.sub("[^a-zA-Z0-9]", " ", raw_article)

    # convert words to lower case and split
    words = article_text.lower().split()

    # remove stopwords
    if remove_stopwords is True:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return words



def create_tfidf_model(articles):
    # convert everything to wordlists
    texts = []
    i = 0
    for article in articles:
        i += 1
        texts.append(article_to_wordlist(article, True, True))
        if i % 1000 == 0:
            print("processing document: ", i)

    # remove words that appear only once
    print("counting word frequencies")
    frequencies = defaultdict(int)
    for text in texts:
        for word in text:
            frequencies[word] += 1

    print("removing words that appear only once")
    texts = [[word for word in text if frequencies[word] > 1] for text in texts]

    # convert to corpus
    print("creating dictionary")
    dct = corpora.Dictionary(texts)
    print("converting to corpus")
    corpus = [dct.doc2bow(line) for line in texts]

    # fit model
    print("fitting tfidf model")
    model = models.TfidfModel(corpus)

    return model, dct, corpus


MODEL_SAVE_FILEPATH = "/storage/nlp/tf_idf_news.idf"
DICT_SAVE_FILEPATH = "/storage/nlp/tf_idf_news.dict"
CORPUS_SAVE_FILEPATH = "/storage/nlp/tf_idf_news.corp"

def create_and_save_tfidf():
    documents = get_documents()
    model, dictionary, corpus = create_tfidf_model(documents)

    model.save(MODEL_SAVE_FILEPATH)
    dictionary.save(DICT_SAVE_FILEPATH)
    corpora.MmCorpus.serialize(CORPUS_SAVE_FILEPATH, corpus)


def load_tfidf_model():
    model = models.TfidfModel.load(MODEL_SAVE_FILEPATH)
    dictionary = corpora.Dictionary.load(DICT_SAVE_FILEPATH)
    corpus = corpora.MmCorpus(CORPUS_SAVE_FILEPATH)

    return model, dictionary, corpus



if __name__=="__main__":
    create_and_save_tfidf()
