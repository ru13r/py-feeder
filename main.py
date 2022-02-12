# utilities
from funcy import *
from sty import ef, fg

# feed parser
import feedparser

# NLP and Clustering
import spacy
from sklearn.cluster import SpectralClustering


# currying lib functions for composition
walk = curry(walk)
lpluck = curry(lpluck)
lfilter = curry(lfilter)
lpluck_attr = curry(lpluck_attr)
take = curry(take)
group_by = curry(group_by)

# load russian language model
nlp = spacy.load("ru_core_news_md")


# List of feeds URLs -> List of Entries
load_rss = compose(
    join,
    lambda feeds: [feedparser.parse(feed).entries for feed in feeds]
)


# select attributes from the Entries
@curry
def select(attributes, entries):
    return [{key: getattr(entry, key, '')
             for key in attributes}
            for entry in entries]


# Tag each entry with 'keywords'
# String -> [String]
def add_keywords(entry):
    # Get list of unique lemmas for the string
    # String -> [String]
    get_keywords = compose(
        ' '.join,
        distinct,
        lpluck_attr('lemma_'),
        lfilter(lambda t: t.pos_ in ['PROPN', 'NOUN']),
        lfilter(lambda t: not (t.is_stop | t.is_punct)),
        lambda doc: [token for token in doc],
        nlp,
    )
    return dict(entry, **{'keywords': get_keywords(entry['title'])})


# tag each entry with 'cluster' number
# [Entries] -> [Entries]
def clusterize(entries):
    clusters = compose(
        SpectralClustering(80).fit_predict,
        lambda xs: [[x1.similarity(x2) for x1 in xs] for x2 in xs],
        walk(nlp),
        lpluck('keywords')
    )
    return [dict(d, **{'cluster': v}) for v, d in zip(clusters(entries), entries)]


# Printing entry titles to console
# Entry -> String
def prettify(entry):
    return fg.blue + '[' + str(entry['cluster']) + '] ' + fg.rs + \
        ef.b + entry['title'] + ef.rs
    # + '\n' + ef.dim + fill(entry['summary']) + ef.dim
    # + '\n' + '[' + ef.dim + entry['keywords'] + ef.dim + ']'


# Prepare a printable string
# [Entries] -> String
print_entries = compose(
    '\n'.join,
    walk(prettify)
)

print_clusters = compose(
    '\n\n'.join,
    walk(print_entries),
    lambda d: d.values()
)


def main():
    feeds = [
        'https://lenta.ru/rss/news',
        'https://news.ru/rss/type/post/',
        'https://ria.ru/export/rss2/archive/index.xml',
        'https://www.vedomosti.ru/rss/news',
        'https://russian.rt.com/rss'
    ]
    output = compose(
        print_clusters,
        group_by(lambda x: x['cluster']),
        clusterize,
        walk(add_keywords),
        select(['title']),
        load_rss
    )(feeds)
    print(output)


if __name__ == '__main__':
    main()
