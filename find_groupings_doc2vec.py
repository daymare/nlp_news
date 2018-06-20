"""

goal: group articles into groups of articles with the same topic
method: using word vector sum and cosine similarity as distance metric between two article titles

"""

from nltk.corpus import stopwords

import gensim
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

import multiprocessing
import numpy as np

from random import shuffle

import read_data
import utility
import pickle

# load up number of cores and ensure gensime will not be painfully slow
cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1

# constants
alpha = 0.025
min_alpha = 0.001
passes = 20
alpha_delta = (alpha - min_alpha) / passes


def main():
    # get articles
    articles = read_data.get_documents_w_metadata()

    # convert to wordlists
    wordlists = read_data.articles_to_wordlist(articles)

    # setup tagged documents
    tagged_docs = []
    for i in range(len(wordlists)):
        words = wordlists[i]
        tags = [i]
        tagged_docs = TaggedDocument(words, tags)

    # build base models
    base_models = \
    [
        # PV-DM w/concatenation
        Doc2Vec(dm=1, dm_concat=1, size=100, window=5, 
                negative=5, hs=0, min_count=2, workers=cores),
        # PV-DBOW
        Doc2Vec(dm=0, size=100, negative=5, hs=0, 
                min_count=2, workers=cores),
        # PV-DM w/average
        Doc2Vec(dm=1, size=100, window=10, negative=5, 
                hs=0, min_count=2, workers=cores)
    ]

    # setup wordlist
    base_models[0].build_vocab(wordlists)

    print(base_models[0])
    for model in base_models[1:]:
        model.reset_from(base_models[0])
        print(model)

    # setup models dictionary
    models_by_name = {}
    models_by_name['dmc'] = base_models[0]
    models_by_name['dbow'] = base_models[1]
    models_by_name['dmm'] = base_models[2]

    # set up more complicated models
    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec(
            [models_by_name['dbow'], models_by_name['dmm']])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec(
            [models_by_name['dbow'], models_by_name['dmc']])

    # train models
    for epoch in range(passes):
        shuffle(tagged_docs)

        for name, train_model in models_by_name.items():
            # train
            train_model.alpha, train_model.min_alpha = alpha, alpha
            train_model.train(tagged_docs, total_examples=len(tagged_docs), epochs=1)

            # evaluate

        # decay alpha
        print('Completed pass %i at alpha %f' % (epoch+1, alpha))
        alpha -= alpha_delta




if __name__ == "__main__":
    main()
