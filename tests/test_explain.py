import pandas as pd
import numpy as np

from matching.vectorizer import FeatureVectorizer
from matching.matcher import MatchingEngine
from matching.explain import explain_matches


def test_explain_matches_basic():
    df = pd.DataFrame({
        "age": [30, 50, 40],
        "erythema": [0.2, 0.8, 0.5],
        "sex": ["F", "M", "F"],
        "biologic": ["Dupixent", "Dupixent", "Ebglyss"],
        "outcome": [1, 0, 1],
        "case_id": ["c0", "c1", "c2"],
    })

    num = ["age", "erythema"]
    cat = ["sex", "biologic"]

    v = FeatureVectorizer(numeric_features=num, categorical_features=cat)
    X = v.fit_transform(df)

    engine = MatchingEngine(metric="cosine")
    engine.fit(X, df[["biologic", "outcome", "case_id"]])

    q = X[0]
    ex = explain_matches(v, engine, q, k=2, top_n=2)
    assert isinstance(ex, list)
    assert len(ex) == 2
    for e in ex:
        assert "top_features" in e
        assert len(e["top_features"]) <= 2
        # feature names should come from vectorizer
        names = [t[0] for t in e["top_features"]]
        for n in names:
            assert n in v.get_feature_names()
