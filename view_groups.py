
import pickle

import read_data


def main():
    # load up the groups
    group_file = open("groups.pickle", "rb")
    groups = pickle.load(group_file)

    # load up the articles
    articles = read_data.get_documents_w_metadata()

    # display each group
    for i in range(len(groups)):
        print("group: %d" % i)
        
        group = groups[i]
        for article_id in group:
            # get article
            article = articles[article_id]

            # print article information
            print()
            print("article: " + article.title)
            print("publication: " + article.publication)
            print("date: " + str(article.date))
            print("content: " + article.content[:200] + "...")

        print()
        print("===================================")
        print()

if __name__=="__main__":
    main()

