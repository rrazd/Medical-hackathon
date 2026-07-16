"""Bias analysis: synthetic erythema sweep to surface sensitivity to skin-tone proxies.

Creates a small reference dataset, initializes the demo matcher, then for a canonical
patient sweeps the 'erythema' input across a range and records per-biologic likelihoods.
Writes results to data/bias_sweep.csv and a short Markdown report in .planning/phases/04-Similar-Patient-Matching/Bias-Report.md
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import pandas as pd

from matching.vectorizer import FeatureVectorizer
from matching.matcher import MatchingEngine
from matching.scoring import calibrate_results

OUT_DIR = Path(".planning/phases/04-Similar-Patient-Matching")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_sweep():
    ref = pd.DataFrame({
        "age": [30, 50, 40, 60],
        "erythema": [0.3, 0.85, 0.45, 0.9],
        "sex": ["F", "M", "F", "M"],
        "biologic": ["Dupixent", "Dupixent", "Ebglyss", "Dupixent"],
        "outcome": [1, 0, 1, 1],
        "case_id": ["c0", "c1", "c2", "c3"],
    })

    numeric = ["age", "erythema"]
    cat = ["sex", "biologic"]

    vect = FeatureVectorizer(numeric_features=numeric, categorical_features=cat)
    X = vect.fit_transform(ref)
    engine = MatchingEngine(metric="cosine")
    engine.fit(X, ref[["biologic", "outcome", "case_id"]])

    # canonical patient baseline (age 42, sex F)
    base = {"age": 42, "sex": "F"}

    sweep = np.linspace(0.05, 0.95, 19)
    rows = []
    for e in sweep:
        row = pd.DataFrame([{"age": base["age"], "erythema": float(e), "sex": base["sex"], "biologic": ""}])
        vec = vect.transform(row)[0]
        raw = engine.compute_likelihoods(vec, k=3)
        # build neighbor_weights_map for calibration
        idxs, dists = engine.kneighbors(vec, k=3)
        weights = 1.0 / (dists + 1e-6)
        rows_meta = engine._meta.iloc[idxs]
        neighbor_weights_map = {}
        for pos, biologic in enumerate(rows_meta["biologic"].values):
            neighbor_weights_map.setdefault(biologic, []).append(float(weights[pos]))
        calibrated = calibrate_results(raw, neighbor_weights_map)
        dup = calibrated.get("Dupixent", {}).get("p_hat", None)
        ebg = calibrated.get("Ebglyss", {}).get("p_hat", None)
        rows.append({"erythema": float(e), "Dupixent_p_hat": dup, "Ebglyss_p_hat": ebg})

    csv_path = OUT_DIR / "bias_sweep.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["erythema", "Dupixent_p_hat", "Ebglyss_p_hat"])
        writer.writeheader()
        writer.writerows(rows)

    # generate a brief markdown report summarizing sensitivity
    report_path = OUT_DIR / "Bias-Report.md"
    with report_path.open("w") as f:
        f.write("# Bias Analysis — Erythema Sweep\n\n")
        f.write("This synthetic sweep varies the erythema input for a canonical patient (age 42, sex=F) and records per-biologic probability estimates from the matching engine. This is a demo-level sensitivity check, not clinical validation.\n\n")
        f.write("## Summary\n\n")
        f.write("Results written to bias_sweep.csv. Observations:\n\n")
        # simple observation: compute min/max and range
        dups = [r["Dupixent_p_hat"] for r in rows if r["Dupixent_p_hat"] is not None]
        ebgs = [r["Ebglyss_p_hat"] for r in rows if r["Ebglyss_p_hat"] is not None]
        if dups:
            f.write(f"- Dupixent p_hat range: {min(dups):.3f} → {max(dups):.3f}\n")
        if ebgs:
            f.write(f"- Ebglyss p_hat range: {min(ebgs):.3f} → {max(ebgs):.3f}\n")
        f.write("\n## Mitigations (demo-level)\n\n")
        f.write("- Display small-sample confidence and matched-case examples rather than sole reliance on a point estimate.\n")
        f.write("- Use biomarkers less sensitive to skin tone (texture/scaling) as alternative signals.\n")
        f.write("- Increase reference dataset diversity and re-run sweep across real skin-tone variants.\n")

    return csv_path, report_path


if __name__ == '__main__':
    csvp, rp = run_sweep()
    print(f"Wrote {csvp} and {rp}")
