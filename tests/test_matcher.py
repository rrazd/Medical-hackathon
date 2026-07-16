import numpy as np
import pandas as pd

from matching.vectorizer import FeatureVectorizer
from matching.matcher import MatchingEngine


def test_matching_engine_basic():
    # Create small dataset where outcome=1 indicates improvement
    df = pd.DataFrame(
        {
            "age": [30, 50, 40, 60],
            "erythema": [0.2, 0.8, 0.4, 0.9],
            "sex": ["F", "M", "F", "M"],
            "biologic": ["Dupixent", "Dupixent", "Ebglyss", "Dupixent"],
            "outcome": [1, 0, 1, 1],
        }
    )

    num = ["age", "erythema"]
    cat = ["sex", "biologic"]

    v = FeatureVectorizer(numeric_features=num, categorical_features=cat)
    X = v.fit_transform(df)

    engine = MatchingEngine(metric="cosine")
    engine.fit(X, df[["biologic", "outcome"]])

    # Query using the first row vector — expect biologic 'Dupixent' to appear
    q = X[0]
    results = engine.compute_likelihoods(q, k=3)

    assert isinstance(results, dict)
    # Expect at least one biologic present
    assert len(results) >= 1
    # Likelihood values between 0 and 1
    for b, info in results.items():
        assert 0.0 <= info["likelihood"] <= 1.0
        assert info["support_count"] >= 1

    matches = engine.get_top_matches(q, k=3)
    assert isinstance(matches, list)
    assert len(matches) == 3 or len(matches) == min(3, len(df))
    # top match should include biologic key
    assert "biologic" in matches[0]
