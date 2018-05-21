"""

goal: group articles into groups of articles with the same topic

"""

import read_data
import utility
import pickle

from gensim import similarities

# constants
alpha = 4
grouping_cutoff = 0.15


def main():
    model, dictionary, corpus = read_data.load_tfidf_model()

    # get articles
    #articles = read_data.get_documents()
    articles = read_data.get_documents_w_metadata()

    # create similarity index
    print("create similarity index")

    nf = len(dictionary.dfs)
    index = similarities.SparseMatrixSimilarity(model[corpus], num_features=nf)

    # create set of all articles
    ungrouped_set = set()
    for i in range(len(articles)):
        ungrouped_set.add(i)

    groups = []
    while bool(ungrouped_set) is True and len(groups) < 500:
        # find the next group
        next_group = find_group(ungrouped_set, articles, index, model, dictionary)
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


def find_group(ungrouped_set, articles_list, index, model, dictionary):
    """
        find a grouping of similar articles

        inputs:
            ungrouped_set: set of all ungrouped article numbers
            articles_list: list of all articles
            index: tf-idf index to do similarity query on
            model: tf-idf model
    """
    # get first article out of set
    article_id = next(iter(ungrouped_set))

    # prepare article for querying
    article = articles_list[article_id]
    article_wordlist = read_data.article_to_wordlist(article.content)
    article_bow = dictionary.doc2bow(article_wordlist)
    article_tfidf = model[article_bow]

    # perform query on article
    similarity_query = index[article_tfidf]

    # add date modifier
    # score = similarity_score * (alpha / (alpha + days apart))
    for i in range(len(similarity_query)):
        similarity_score = similarity_query[i]
        date_distance = utility.get_date_distance(article, articles_list[i])
        score = similarity_score * (alpha / (alpha + date_distance))
        similarity_query[i] = score

    # sort by score and enumerate each document
    # result is a list of tuples: (article_id, score)
    similarity_query = sorted(enumerate(similarity_query), key=lambda item:-item[1])

    # find group members
    group = []
    for i in range(len(similarity_query)):
        if similarity_query[i][1] > grouping_cutoff:
            # article is similar enough
            # note that the first article in the sorted similarity should be the queried document.
            group.append(similarity_query[i][0])
        else:
            # since the list is sorted no other articles can be above the grouping cutoff
            break

    return group

if __name__ == "__main__":
    main()
