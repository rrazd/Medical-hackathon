"""Scoring utilities for per-biologic likelihood and uncertainty.

Converts MatchingEngine weighted aggregates into calibrated likelihood estimates,
computes effective sample size (Kish), and returns Wilson score confidence
intervals to convey small-sample uncertainty.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple


def effective_sample_size(weights: list[float]) -> float:
    """Compute Kish effective sample size: (sum w)^2 / sum(w^2)."""
    s = sum(weights)
    ssq = sum(w * w for w in weights)
    if ssq <= 0:
        return 0.0
    return (s * s) / ssq


def wilson_interval(p_hat: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for proportion p_hat with sample size n.

    If n <= 0, returns (0.0, 1.0) to reflect complete uncertainty.
    """
    if n <= 0:
        return 0.0, 1.0
    denom = 1 + (z * z) / n
    center = p_hat + (z * z) / (2 * n)
    margin = z * math.sqrt((p_hat * (1 - p_hat) + (z * z) / (4 * n)) / n)
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    return lower, upper


def calibrate_results(raw_results: Dict[str, Dict], neighbor_weights_map: Dict[str, list[float]]) -> Dict[str, Dict]:
    """Take MatchingEngine.compute_likelihoods raw output and enrich with:
    - p_hat: weighted proportion
    - effective_n: Kish effective sample size computed from per-match weights
    - ci_lower, ci_upper: Wilson interval using effective_n

    neighbor_weights_map: mapping biologic -> list of weights for the matched neighbors
    (weights should correspond to the neighbors that contributed to that biologic).
    """
    out: Dict[str, Dict] = {}
    for biologic, info in raw_results.items():
        weighted_sum = info.get("weighted_sum", 0.0)
        weight_total = info.get("weight_total", 0.0)
        # p_hat is weighted_sum / weight_total if weight_total > 0
        p_hat = (weighted_sum / weight_total) if weight_total > 0 else 0.0
        weights = neighbor_weights_map.get(biologic, [])
        eff_n = effective_sample_size(weights)
        ci_lower, ci_upper = wilson_interval(p_hat, eff_n)
        enriched = dict(info)
        enriched.update({"p_hat": p_hat, "effective_n": eff_n, "ci_lower": ci_lower, "ci_upper": ci_upper})
        out[biologic] = enriched
    return out


__all__ = ["effective_sample_size", "wilson_interval", "calibrate_results"]
