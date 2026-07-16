"""Nearest-neighbor matching engine for Similar-Patient Matching.

Provides a thin wrapper over sklearn.neighbors.NearestNeighbors and helpers to
compute per-biologic likelihoods using distance-weighted averaging with
Laplace smoothing for small-sample stability.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.neighbors import NearestNeighbors
    from joblib import dump, load
except Exception as e:  # pragma: no cover - dependency error
    raise ImportError("scikit-learn and joblib are required for matcher") from e


class MatchingEngine:
    def __init__(self, metric: str = "cosine") -> None:
        self.metric = metric
        self._nn: Optional[NearestNeighbors] = None
        self._X: Optional[np.ndarray] = None
        self._meta: Optional[pd.DataFrame] = None

    def fit(self, X: np.ndarray, metadata: pd.DataFrame) -> "MatchingEngine":
        """Fit the internal nearest-neighbor index.

        X: numeric matrix shape (n_samples, n_features)
        metadata: DataFrame with n_samples rows containing at least 'biologic' and 'outcome'
        """
        if X.shape[0] != len(metadata):
            raise ValueError("X rows must match metadata length")
        self._X = np.asarray(X)
        self._meta = metadata.reset_index(drop=True)
        self._nn = NearestNeighbors(metric=self.metric)
        self._nn.fit(self._X)
        return self

    def kneighbors(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self._nn is None:
            raise RuntimeError("MatchingEngine must be fitted before kneighbors()")
        q = np.asarray(query).reshape(1, -1)
        dists, idxs = self._nn.kneighbors(q, n_neighbors=min(k, len(self._X)))
        return idxs.ravel(), dists.ravel()

    def _weights_from_distances(self, dists: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        # Convert distances to positive weights; small eps to avoid div-by-zero
        return 1.0 / (dists + eps)

    def compute_likelihoods(self, query: np.ndarray, k: int = 5, alpha: float = 1.0) -> Dict[str, Dict]:
        """Compute per-biologic likelihoods of positive outcome for a query.

        Returns a dict keyed by biologic with {likelihood, support_count, weighted_sum, weight_total}.

        Uses Laplace smoothing parameter alpha.
        """
        if self._X is None or self._meta is None:
            raise RuntimeError("Engine not fitted")
        idxs, dists = self.kneighbors(query, k=k)
        weights = self._weights_from_distances(dists)

        rows = self._meta.iloc[idxs]
        results: Dict[str, Dict] = {}

        for biologic, grp in rows.groupby("biologic"):
            # align weights to group rows
            grp_idxs = grp.index.values
            # find positions of grp_idxs within idxs array (robust to numpy scalar conversion)
            idxs_list = idxs.tolist()
            positions = [idxs_list.index(int(i)) for i in grp_idxs]
            w = weights[positions]
            outcomes = grp["outcome"].astype(float).values
            weighted_sum = float((w * outcomes).sum())
            weight_total = float(w.sum())
            # Laplace smoothing with alpha over a Bernoulli outcome
            likelihood = (weighted_sum + alpha * 0.5) / (weight_total + alpha)
            results[biologic] = {
                "likelihood": likelihood,
                "support_count": int(len(grp)),
                "weighted_sum": weighted_sum,
                "weight_total": weight_total,
            }

        return results

    def get_top_matches(self, query: np.ndarray, k: int = 5) -> List[Dict]:
        idxs, dists = self.kneighbors(query, k=k)
        rows = self._meta.iloc[idxs].copy()
        rows = rows.reset_index(drop=True)
        matches = []
        for i, (idx, dist) in enumerate(zip(idxs, dists)):
            m = rows.iloc[i].to_dict()
            m.update({"_index": int(idx), "distance": float(dist)})
            matches.append(m)
        return matches

    def save(self, path: str | Path) -> None:
        if self._X is None or self._meta is None or self._nn is None:
            raise RuntimeError("Fit the engine before saving")
        dump({"nn": self._nn, "X": self._X, "meta": self._meta}, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "MatchingEngine":
        data = load(Path(path))
        obj = cls()
        obj._nn = data["nn"]
        obj._X = data["X"]
        obj._meta = data["meta"]
        return obj


__all__ = ["MatchingEngine"]
