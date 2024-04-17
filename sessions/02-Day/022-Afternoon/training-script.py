"""
"""

## Imports
##################################################

import os, sys, warnings, secrets, datetime
import pickle
from joblib import dump, load

# from IPython.display import display

import pandas as pd

# import numpy as np

from sklearn.base import *
from sklearn.preprocessing import *
from sklearn.impute import *
from sklearn.model_selection import *
from sklearn.decomposition import *
from sklearn.ensemble import *
from sklearn.model_selection import *
from sklearn.pipeline import *
from sklearn.feature_extraction import *
from sklearn.dummy import *
from sklearn.feature_extraction.text import *

from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.neighbors import *

# import nltk
# from nltk.corpus import stopwords
# from nltk.corpus import words
# from nltk.tokenize import wordpunct_tokenize

# import string

import spacy

# from spacy.lang.en.stop_words import STOP_WORDS


# import gensim

from gensim.models import KeyedVectors
from gensim.downloader import load

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import preprocess_string

# from openai import OpenAI

# import requests

warnings.filterwarnings("ignore")


## Functions
##################################################


def resultize(grid, head=20, export=False, token=None):

    res = pd.DataFrame(grid.cv_results_)
    cols = [i for i in res.columns if "split" not in i]
    res = res.loc[:, cols]
    res = res.round(2).sort_values("mean_test_score", ascending=False).head(head)

    if export:
        _res = res.copy()
        _res["token"] = token
        _res = _res.astype(str)
        now = str(datetime.datetime.now())[:19].replace(" ", "_")
        _res.to_csv(f"results__{token}__{now}.csv", index=False)

    return res


def save(model, base_fn, token=None, score=None):
    """Saving our model"""

    if not token:
        token = secrets.token_hex(4)

    now = str(datetime.datetime.now())[:19].replace(" ", "_")

    fn = f"{base_fn}__{token}__{now}"
    if score:
        fn += "__" + str(round(score, 2))

    dump(model, fn)

    return fn, sys.getsizeof(model)


def load(fn):

    if not os.path.isfile(fn):
        raise AttributeError(f"The File {fn} do not exists!")

    model = load(fn)

    return model, sys.getsizeof(model)


def cv():
    return StratifiedShuffleSplit(n_splits=5, test_size=0.25)


## Classes
##################################################


class Doc2VecTransformer(BaseEstimator, TransformerMixin):
    """ """

    def __init__(
        self,
        model=None,
        vector_size=500,
        window=5,
        min_count=5,
        epochs=100,
    ):

        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = model

    def fit(self, X, y=None):

        if not isinstance(X, list):
            _X = X.values.tolist()
        else:
            _X = X

        if self.model:
            return self

        tagged_docs = [
            TaggedDocument(words=preprocess_string(doc), tags=[i])
            for i, doc in enumerate(_X)
        ]
        model = Doc2Vec(
            vector_size=self.vector_size, min_count=self.min_count, epochs=self.epochs
        )
        model.build_vocab(tagged_docs)
        model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
        self.model = model

        return self

    def transform(self, X, y=None):

        if not isinstance(X, list):
            _X = X.values.tolist()
        else:
            _X = X

        vectors = [self.model.infer_vector(preprocess_string(i)) for i in X]
        return vectors


## Values & Vars and Consts
##################################################


pipeline = Pipeline(
    [
        ("preprocessor", "passthrough"),
        ("scaler", "passthrough"),
        ("reductor", "passthrough"),
        ("estimator", LogisticRegression()),
    ]
)


pst = "passthrough"


nlp = spacy.load("en_core_web_lg")

url = "https://gist.githubusercontent.com/AlexandreGazagnes/cabe445634a092d308d17a883a305a75/raw/d2014e8a34bba3c1be3ec8936bb338fb42888f24/nlp.csv"


## Data
##################################################


df = pd.read_csv(url)
# df.head(5)


y = df.cat_1
y


X = df.description
X


## Param grid
##################################################

# param_grid = {
#     "scaler": [
#         "passthrough",
#         StandardScaler(),
#         QuantileTransformer(n_quantiles=100),
#         # MinMaxScaler(),
#         Normalizer(),
#     ],
#     "reductor": [PCA()],
#     "reductor__n_components": [0.7, 0.85, 0.9, 0.95, 0.99],
#     "estimator": [RandomForestClassifier(), LogisticRegression()],
# }

# param_grid = {
#     "preprocessor": [Doc2VecTransformer()],
#     "estimator": [RandomForestClassifier(), LogisticRegression()],
# }


param_grid = {
    "preprocessor": [Doc2VecTransformer()],
    "preprocessor__vector_size": [500],  # 100, 200, # 1000
    "preprocessor__window": [5],  # 5, 10, 5, 10,   # 20, 25, 30
    "preprocessor__epochs": [500],
    "preprocessor__min_count": [3],  # 100, 300,  # 1000
    "estimator": [LogisticRegression()],  # RandomForestClassifier()
    # preprocessor__model = [model_sm, model_md, model_lg, model_xl]
}

# param_grid = {
#     "preprocessor__window": [5, 10, 15],
#     # preprocessor__model = [model_sm, # param_grid = {
#     "preproce
#     ],  # 100, 200, # 1000
#     "prepr
#     ],  # 5, 10, 5, 10,   # 20, 25, 30
#     "prep00
#     ],
#     "preprocessor__min_count": [1, 3, 5],  # 100, 300,  # 1000
#     ## param_grid = {
#     "preprocessor__window": [10, 15],  # 5
#     # preprocessor__model = [model_sm,


## GridSearchCV
##################################################

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,  # Just for a try
    n_jobs=6,  # -1 is good
    return_train_score=True,
    verbose=2,
    refit=True,
)


# prints
print(pipeline)
print(param_grid)
print(grid)


# fit
grid.fit(df.description, df.cat_1)
print(grid.best_estimator_)

res = resultize(grid, head=30, export=True)
print(res.head(1))

# best pipeline
best_pipeline = grid.best_estimator_
print(best_pipeline)


# Save
out = save(best_pipeline, "gensim_pipe")
out = save(Doc2VecTransformer, "d2v_transformer")
print(str(out[0]))
print(str(out[1]))

# Out
out = best_pipeline.predict(["i have a beautiful analogic watch rolex"])
print(str(out))

# grid.predict(["i have a beautiful analogic watch rolex"])
