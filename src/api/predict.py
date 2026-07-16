"""Simple FastAPI predict endpoint wiring the extractor -> vectorizer -> matcher -> scorer flow.

This module is intentionally minimal: it defines Pydantic request/response models and
an endpoint `/api/predict` that accepts patient features and (optionally) a precomputed
biomarker vector. It uses the FeatureVectorizer and MatchingEngine created earlier.

Note: requires FastAPI and uvicorn to run; tests and full app wiring are out-of-scope here.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

from matching.vectorizer import FeatureVectorizer
from matching.matcher import MatchingEngine
from matching.scoring import calibrate_results

app = FastAPI()


class PatientFeatures(BaseModel):
    age: float
    erythema: float
    sex: str
    biologic: Optional[str] = None


class MatchCase(BaseModel):
    case_id: str
    distance: float
    biologic: str
    outcome: float


class PredictResponse(BaseModel):
    likelihoods: Dict[str, Dict]
    matches: List[MatchCase]


# NOTE: in a real app these would be application-scoped singletons loaded at startup
_VECTORIZER: Optional[FeatureVectorizer] = None
_ENGINE: Optional[MatchingEngine] = None


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PatientFeatures, top_k: int = 5):
    global _VECTORIZER, _ENGINE
    if _VECTORIZER is None or _ENGINE is None:
        raise HTTPException(status_code=503, detail="Matching service not initialized")

    # Build dataframe row for vectorizer
    # Ensure row includes any categorical columns the vectorizer expects (use empty string defaults)
    row_dict = {"age": payload.age, "erythema": payload.erythema, "sex": payload.sex}
    # If vectorizer knows categorical features, pre-populate with empty values to avoid missing columns
    if _VECTORIZER is not None:
        for c in _VECTORIZER.categorical_features:
            if c not in row_dict:
                row_dict[c] = ""
    row = pd.DataFrame([row_dict])

    try:
        vec = _VECTORIZER.transform(row)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Vectorization failed: {e}")

    # compute matches and likelihoods
    matches = _ENGINE.get_top_matches(vec, k=top_k)
    raw = _ENGINE.compute_likelihoods(vec, k=top_k)

    # Build neighbor_weights_map needed by calibrate_results:
    # For simplicity, re-run kneighbors to obtain distances/weights per neighbor and map them per biologic
    idxs, dists = _ENGINE.kneighbors(vec, k=top_k)
    weights = 1.0 / (dists + 1e-6)
    rows = _ENGINE._meta.iloc[idxs]
    neighbor_weights_map: Dict[str, List[float]] = {}
    for pos, biologic in enumerate(rows["biologic"].values):
        neighbor_weights_map.setdefault(biologic, []).append(float(weights[pos]))

    calibrated = calibrate_results(raw, neighbor_weights_map)

    match_items = [
        MatchCase(
            case_id=str(m.get("case_id", m.get("_index"))),
            distance=m.get("distance", 0.0),
            biologic=m.get("biologic", ""),
            outcome=float(m.get("outcome", 0.0)),
        )
        for m in matches
    ]

    return PredictResponse(likelihoods=calibrated, matches=match_items)


# Helper to initialize the in-memory matcher for demo/testing
def initialize_demo(reference_df: pd.DataFrame, numeric: List[str], categorical: List[str]):
    global _VECTORIZER, _ENGINE
    _VECTORIZER = FeatureVectorizer(numeric_features=numeric, categorical_features=categorical)
    X = _VECTORIZER.fit_transform(reference_df)
    engine = MatchingEngine(metric="cosine")
    engine.fit(X, reference_df[["biologic", "outcome"]].copy())
    _ENGINE = engine
    return _VECTORIZER, _ENGINE


__all__ = ["app", "initialize_demo", "PatientFeatures", "PredictResponse"]
