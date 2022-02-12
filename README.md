# py-feeder
A simple RSS parser written in Python 
using NLP model to detect and group similar news.

Libraries used:
* [spacy](https://spacy.io/) for lemmatization
* [scikit-learn](https://scikit-learn.org/stable/index.html) for clustering

### Installation
Clone the repository.
```commandline
git clone https://github.com/ru13r/py-feeder
```

Create the virtual environment, then run
```commandline
pip3 install
```

For [spacy](https://spacy.io/) model to work you also need 
to manually download the model file
```commandline
python -m spacy download ru_core_news_md
```