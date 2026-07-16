"""Feature vectorizer for Similar-Patient Matching.

Implements a sklearn ColumnTransformer-based pipeline to normalize numeric
features and one-hot encode categorical features. Provides fit/transform,
feature-name export, and joblib save/load helpers.

Requires: pandas, numpy, scikit-learn, joblib
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from joblib import dump, load
except Exception as e:  # pragma: no cover - explicit dependency error
    raise ImportError(
        "Missing dependency for matching vectorizer: scikit-learn and joblib are required"
    ) from e


class FeatureVectorizer:
    """Builds and manages a ColumnTransformer pipeline.

    Args:
        numeric_features: list of numeric column names
        categorical_features: list of categorical column names
    """

    def __init__(
        self,
        numeric_features: List[str],
        categorical_features: Optional[List[str]] = None,
    ) -> None:
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features or [])
        self._fitted = False

        num_pipeline = Pipeline([("scaler", StandardScaler())])
        cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        transformers = []
        if self.numeric_features:
            transformers.append(("num", num_pipeline, self.numeric_features))
        if self.categorical_features:
            transformers.append(("cat", cat_encoder, self.categorical_features))

        self._ct = ColumnTransformer(transformers=transformers, remainder="drop")

    def fit(self, df: pd.DataFrame) -> "FeatureVectorizer":
        """Fit the transformer to a dataframe and return self."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        # ensure expected columns exist
        missing = [c for c in self.numeric_features + self.categorical_features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in input DataFrame: {missing}")
        self._ct.fit(df)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a dataframe into a numeric feature matrix."""
        if not self._fitted:
            raise RuntimeError("FeatureVectorizer must be fitted before transform()")
        return self._ct.transform(df)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit then transform."""
        self.fit(df)
        return self.transform(df)

    def get_feature_names(self) -> List[str]:
        """Return output feature names in the same order as transform() columns."""
        if not self._fitted:
            raise RuntimeError("FeatureVectorizer must be fitted to get feature names")
        # ColumnTransformer exposes get_feature_names_out in sklearn>=1.0
        try:
            names = list(self._ct.get_feature_names_out())
        except Exception:
            # Fallback: attempt to construct names manually
            names: List[str] = []
            if self.numeric_features:
                names.extend(self.numeric_features)
            if self.categorical_features:
                # get categories from fitted OneHotEncoder
                enc = None
                for name, trans, cols in self._ct.transformers:
                    if name == "cat":
                        enc = trans
                        break
                if enc is not None and hasattr(enc, "categories_"):
                    for col, cats in zip(self.categorical_features, enc.categories_):
                        names.extend([f"{col}__{v}" for v in cats])
        return names

    def save(self, path: str | Path) -> None:
        """Persist the fitted ColumnTransformer to disk via joblib."""
        if not self._fitted:
            raise RuntimeError("Fit before saving the vectorizer")
        dump(self._ct, Path(path))

    @classmethod
    def load(cls, path: str | Path, numeric_features: List[str], categorical_features: Optional[List[str]] = None) -> "FeatureVectorizer":
        """Load a persisted ColumnTransformer and wrap it in a FeatureVectorizer shell."""
        obj = cls(numeric_features=numeric_features, categorical_features=categorical_features)
        obj._ct = load(Path(path))
        obj._fitted = True
        return obj


__all__ = ["FeatureVectorizer"]
