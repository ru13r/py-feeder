# utilities
import logging

from funcy import *
from sty import ef, fg
from collections import Counter

# feed parser
import feedparser

# NLP and Clustering
import spacy
from sklearn.cluster import SpectralClustering

# render to html
from jinja2 import Environment, FileSystemLoader
import os

# wordcloud display
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# currying lib functions for composition
walk = curry(walk)
lpluck = curry(lpluck)
lfilter = curry(lfilter)
lpluck_attr = curry(lpluck_attr)
take = curry(take)
group_by = curry(group_by)
lcat = curry(lcat)

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
    custom_stop_words = ['назвать', 'заявить', 'сообщить', 'объявить', 'обсудить', 'называть',
                         'рассказать', 'прокомментировать', 'год', 'месяц', 'призвать']
    get_keywords = compose(
        ' '.join,
        distinct,
        lpluck_attr('lemma_'),
        lfilter(lambda t: not (t.lemma_ in custom_stop_words)),
        lfilter(lambda t: t.pos_ in ['PROPN', 'NOUN', 'VERB']),
        lfilter(lambda t: not (t.is_stop | t.is_punct)),
        lambda doc: [token for token in doc],
        nlp,
    )
    return dict(entry, **{'keywords': get_keywords(entry['title'])})


# tag each entry with 'cluster' number
# [Entries] -> [Entries]
@log_enters(print)
def clusterize(entries):
    clusters = compose(
        SpectralClustering(150).fit_predict,
        lambda xs: [[x1.similarity(x2) for x1 in xs] for x2 in xs],
        walk(nlp),  # TODO optimise
        lpluck('keywords')  # TODO remove double call to lpluck
    )
    return [dict(d, **{'cluster': v}) for v, d in zip(clusters(entries), entries)]


@log_enters(print)
def top_words(entries):
    return compose(
        Counter,
        lambda x: x.split(' '),
        ' '.join,
        lpluck('keywords'),
    )(entries)


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
    lambda d: d.values(),
    group_by(lambda x: x['cluster']),
)


def show_word_cloud(entries):
    text = compose(
        ' '.join,
        # lfilter(lambda w: not (w in ['украина', 'россия'])),
        lambda x: x.split(' '),
        ' '.join,
        lpluck('keywords'),
    )(entries)
    wordcloud = WordCloud(relative_scaling=0.2,
                          background_color='white',
                          width=1600,
                          height=1080,
                          ).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


def render_html(entries):
    clusters = group_by(lambda x: x['cluster'])(entries)

    root = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(root, 'templates')
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template('index.html')
    filename = os.path.join(root, 'output', 'index.html')

    with open(filename, 'w') as fh:
        fh.write(template.render(
            clusters=clusters,
        ))


def main():
    feeds = [
        'https://lenta.ru/rss/news',
        'https://news.ru/rss/type/post/',
        'https://ria.ru/export/rss2/archive/index.xml',
        'https://www.vedomosti.ru/rss/news',
        'https://russian.rt.com/rss'
    ]
    clustered_feeds = compose(
        clusterize,
        walk(add_keywords),
        select(['title', 'link']),
        load_rss
    )(feeds)
    print(print_clusters(clustered_feeds))
    #print(top_words(clustered_feeds))
    #show_word_cloud(clustered_feeds)
    render_html(clustered_feeds)


if __name__ == '__main__':
    main()
