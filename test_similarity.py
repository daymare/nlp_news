"""

goal: test how good the similarity measurements on tf-idf are

"""

import read_data
import utility

from gensim import similarities

# constants
alpha = 4


if __name__ == "__main__":
    model, dictionary, corpus = read_data.load_tfidf_model()

    # get articles
    #articles = read_data.get_documents()
    articles = read_data.get_documents_w_metadata()

    # convert an article to tfidf for querying
    print("convert article to tdifd")
    article_wordlist = read_data.article_to_wordlist(articles[0].content)

    article_bow = dictionary.doc2bow(article_wordlist)
    article_tfidf = model[article_bow]

    # create similarity index
    print("create similarity index")

    nf = len(dictionary.dfs)
    index = similarities.SparseMatrixSimilarity(model[corpus], num_features=nf)

    # perform query
    print("perform query")
    similarity_query = index[article_tfidf]

    # add date modifier
    # goal: penalize articles that are farther away by date
    # score = old_score * (alpha / (alpha + days))
    for i in range(len(similarity_query)):
        current = similarity_query[i]
        
        date_distance = utility.get_date_distance(articles[0], articles[i])
        new_score = current * (alpha / (alpha + date_distance))

        similarity_query[i] = new_score

    print("sorting similarity query")
    similarity_query = sorted(enumerate(similarity_query), key=lambda item: -item[1])

    print("\n")

    for i in range(5):
        article = articles[similarity_query[i][0]]
        print("article %d" % (i + 1))
        print("similarity score: %f" % similarity_query[i][1])
        print("date: ", article.date)
        print("content: ", article.content)
        print("\n")

