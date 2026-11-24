"""Lexical (word + character n-gram) detector. Complements stylometric features:
captures phrasing habits of LLMs that 35 summary statistics cannot."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def build_lexical_model(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("tfidf", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                     lowercase=True, max_features=150_000)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                     sublinear_tf=True, max_features=200_000)),
        ])),
        ("lr", LogisticRegression(C=3.0, max_iter=3000, class_weight="balanced",
                                  random_state=random_state)),
    ])
