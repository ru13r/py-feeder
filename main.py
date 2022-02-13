from funcy import *
from sty import ef, fg

# feed parser
import feedparser

# NLP and Clustering
import spacy
from sklearn.cluster import SpectralClustering

# render to html
from jinja2 import Environment, FileSystemLoader
import os

# wordcloud display
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
nlp = spacy.load("ru_core_news_lg")

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
                         'рассказать', 'прокомментировать', 'год', 'месяц', 'призвать', 'россиянин']
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
@curry
def clusterize(entries):
    max_size = 15   # maximum cluster size
    avg_size = 4    # expected average cluster size

    # group :: [Entries (dicts)] -> defaultDict
    group = group_by(lambda x: x['cluster'])

    # cluster_nums :: [Entries (dicts)] -> [Int32]
    cluster_nums = compose(
        SpectralClustering(len(entries) // avg_size).fit_predict,
        lambda xs: [[x1.similarity(x2) for x1 in xs] for x2 in xs],
        walk(nlp),              # TODO optimise
        lpluck('keywords'),     # TODO remove second call to lpluck
    )

    clusters = group([dict(d, **{'cluster': v}) for v, d in zip(cluster_nums(entries), entries)])
    # iterate with sub-clusters larger than max_size
    large_clusters = {k: v for (k, v) in clusters.items() if len(v) > max_size}
    clusters = {k: v for (k, v) in clusters.items() if not (k in large_clusters)}
    max_key = max(clusters)

    for subcluster in large_clusters.values():
        subcluster = clusterize(subcluster)
        subcluster = {(k + max_key + 1): v for (k, v) in subcluster.items()}
        max_key = max(subcluster)
        clusters = dict(list(clusters.items()) + list(subcluster.items()))

    return clusters


# Printing entry titles to console
# Entry -> String
def prettify(entry):
    return fg.blue + '[' + str(entry['cluster']) + '] ' + fg.rs + \
           ef.b + entry['title'] + ef.rs


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
)


def render_word_cloud(clusters):
    # [Entries] -> String

    text = compose(
        ' '.join,
        lambda x: x.split(' '),
        ' '.join,
        lpluck('keywords'),
        flatten,
    )(clusters.values())
    wordcloud = WordCloud(relative_scaling=0.5,
                          background_color='white',
                          width=750,
                          height=750,
                          ).generate(text)
    return wordcloud.to_svg(embed_font=True)


def render_html(clusters):
    root = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(root, 'templates')
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template('index.html')

    directory = os.path.join(root, 'output')
    file_html = os.path.join(directory, 'index.html')
    file_svg = os.path.join(directory, 'wordcloud.svg')

    svg = render_word_cloud(clusters)
    with open(file_html, 'w') as fh:
        fh.write(template.render(
            clusters=clusters,
        ))
    with open(file_svg, "w+") as fh:
        fh.write(svg)


def main():
    feeds = [
        'https://lenta.ru/rss/news',
        'https://news.ru/rss/type/post/',
        'https://ria.ru/export/rss2/archive/index.xml',
        'https://www.vedomosti.ru/rss/news',
        'https://russian.rt.com/rss',
        'http://static.feed.rbc.ru/rbc/logical/footer/news.rss',
        'https://www.kommersant.ru/RSS/main.xml',
        'https://www.bfm.ru/news.rss?type=news',
    ]
    clustered_feeds = compose(
        clusterize,
        walk(add_keywords),
        select(['title', 'link']),
        load_rss
    )(feeds)
    print(print_clusters(clustered_feeds))
    render_html(clustered_feeds)


if __name__ == '__main__':
    main()
