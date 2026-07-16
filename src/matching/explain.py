"""Explainability helpers for Similar-Patient Matching.

Provides explain_matches(vectorizer, engine, query_vec, k=5, top_n=3) which
returns top contributing features for each of the top-k matched cases.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


def explain_matches(
    vectorizer, engine, query_vec: np.ndarray, k: int = 5, top_n: int = 3
) -> List[Dict]:
    """Return explanations for top-k matches.

    Each explanation dict contains: index, case_id (if available), distance,
    and top_features: list of (feature_name, contribution_score, signed_diff).

    Contribution is approximated as 1/(1+abs(diff)) so smaller absolute differences
    yield higher scores. This is a simple, transparent heuristic suitable for the
    small-data nearest-neighbor approach.
    """
    if engine._X is None or engine._meta is None:
        raise RuntimeError("Engine must be fitted before explaining matches")

    query = np.asarray(query_vec).reshape(-1)
    idxs, dists = engine.kneighbors(query, k=k)
    feature_names = vectorizer.get_feature_names()

    explanations = []
    for pos, (idx, dist) in enumerate(zip(idxs, dists)):
        matched_vec = engine._X[int(idx)]
        diff = query - matched_vec
        absdiff = np.abs(diff)
        # contribution score: higher when absdiff is smaller
        contrib = 1.0 / (1.0 + absdiff)
        # pair feature names with score and signed diff
        paired = [
            (feature_names[i], float(contrib[i]), float(diff[i]))
            for i in range(min(len(feature_names), len(contrib)))
        ]
        # sort by contribution desc
        paired_sorted = sorted(paired, key=lambda x: x[1], reverse=True)
        top = paired_sorted[:top_n]
        meta_row = engine._meta.iloc[np.where(engine._meta.index == idx)[0][0]].to_dict()
        explanations.append(
            {
                "_index": int(idx),
                "case_id": meta_row.get("case_id", meta_row.get("_index", int(idx))),
                "distance": float(dist),
                "top_features": top,
            }
        )
    return explanations


__all__ = ["explain_matches"]
