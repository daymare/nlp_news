"""

goal: group articles into groups of articles with the same topic
method: using word vector sum and cosine similarity as distance metric between two article titles

"""

from nltk.corpus import stopwords
import re
import gensim
import numpy as np
import scipy

import read_data
import utility
import pickle


# constants
alpha = 4
max_groups = 10
grouping_cutoff = 0.20

model = gensim.models.KeyedVectors.load_word2vec_format('/storage/nlp/all_the_news/GoogleNews-vectors-negative300.bin', binary=True)


def main():
    # get articles
    articles = read_data.get_documents_w_metadata()

    # create set of all articles
    ungrouped_set = set()
    for i in range(len(articles)):
        ungrouped_set.add(i)

    groups = []
    while bool(ungrouped_set) is True and len(groups) < max_groups:
        # find the next group
        next_group = find_group(ungrouped_set, articles)
        groups.append(next_group)

        # remove the elements in the next_group
        for article_id in next_group:
            ungrouped_set.discard(article_id)

        print("group: %d with %d articles. %d articles remaining" % (len(groups), len(next_group), len(ungrouped_set)))

    # save groups to file
    print("total number of groups: %d" % len(groups))
    print("saving to pickle")
    save_file = open("./output/groups.pickle", "wb")
    pickle.dump(groups, save_file)


def find_group(ungrouped_set, articles_list):
    """
        find a grouping of similar articles

        inputs:
            ungrouped_set: set of all ungrouped article numbers
            articles_list: list of all articles
    """
    # get first article out of set
    article_id = next(iter(ungrouped_set))
    article = articles_list[article_id]

    # perform query over all articles in the list
    similarity_query = perform_similarity_query(article_id, articles_list)
    if similarity_query is None:
        return [article_id]

    # add date modifier
    # score = similarity_score * (alpha / (alpha + days apart))
    for i in range(len(similarity_query)):
        similarity_score = similarity_query[i]
        date_distance = utility.get_date_distance(article, articles_list[i])
        score = similarity_score * (alpha / (alpha + date_distance))
        similarity_query[i] = score

    # sort by score and enumerate each document
    # result is a list of tuples: (article_id, score)
    similarity_query = sorted(enumerate(similarity_query), key=lambda item:item[1])

    # find group members
    group = []
    for i in range(len(similarity_query)):
        if similarity_query[i][1] < grouping_cutoff:
            # article is similar enough
            # note that the first article in the sorted similarity should be the queried document.
            group.append(similarity_query[i][0])
        else:
            # since the list is sorted no other articles can be above the grouping cutoff
            break

    return group

def perform_similarity_query(article_id, articles_list):
    # convert article title to list of word vectors
    title = articles_list[article_id].title
    print("\ncurrent title: ", title)
    # for some reason titles can be floats
    # check for floats and ignore
    if isinstance(title, float):
        return None

    vectors = string_to_word_vectors(title, model)

    # compare against every article in the list
    similarity_query = []
    for article in articles_list:
        current_title = article.title
        # for some reason titles can be floats
        # check for floats and ignore
        if isinstance(current_title, float):
            similarity_query.append(0.0)
            continue

        current_vectors = string_to_word_vectors(current_title, model)

        current_similarity = get_similarity(vectors, current_vectors)
        if current_similarity < grouping_cutoff:
            print("other title: " + current_title + " similarity: %f" % current_similarity)
        similarity_query.append(current_similarity)

    return similarity_query

def get_similarity(vectors1, vectors2):
    """
        get the similarity between two different lists of word vectors

        We are measuring similarity using the cosine similarity of the
        sums of the lists of word vectors.
    """
    # sum up vectors
    sum1 = np.sum(vectors1, 0)
    sum2 = np.sum(vectors2, 0)

    # get cosine similarity
    return scipy.spatial.distance.cosine(sum1, sum2)

def string_to_word_vectors(string, word_vectors):
    """
        convert a string to it's list of word vectors

        input:
            string - string to be converted to list of word vectors
            word_vectors - global list of word vectors
    """
    # convert to list of words
    wordlist = read_data.article_to_wordlist(string)

    # lookup all word vectors in the wordlist
    wv_list = []
    for word in wordlist:
        if word not in word_vectors.wv:
            continue
        wv = word_vectors.wv[word]
        wv_list.append(wv)

    return wv_list

if __name__ == "__main__":
    main()
