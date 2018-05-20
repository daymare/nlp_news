
from datetime import datetime

# get the date distance between two articles
def get_date_distance(article1, article2):
    if article1.date is None or article2.date is None:
        return 999999
    else:
        delta = article2.date - article1.date
        return abs(delta.days)

