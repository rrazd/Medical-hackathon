import pandas as pd
import numpy as np

from matching.vectorizer import FeatureVectorizer


def test_feature_vectorizer_fit_transform():
    # small synthetic dataframe
    df = pd.DataFrame(
        {
            "age": [25, 60, 42],
            "erythema": [0.2, 0.9, 0.5],
            "sex": ["F", "M", "F"],
            "biologic": ["Dupixent", "Dupixent", "Ebglyss"],
        }
    )

    numeric = ["age", "erythema"]
    categorical = ["sex", "biologic"]

    v = FeatureVectorizer(numeric_features=numeric, categorical_features=categorical)
    out = v.fit_transform(df)

    # ensure numeric matrix returned
    assert isinstance(out, np.ndarray)
    # ensure number of rows matches
    assert out.shape[0] == df.shape[0]

    # feature names are available
    names = v.get_feature_names()
    assert len(names) == out.shape[1]
    # expect age to be present as a numeric feature name
    assert any("age" in n for n in names)
