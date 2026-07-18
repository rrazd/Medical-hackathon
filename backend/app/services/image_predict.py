"""Image-driven similarity matching for the 4-column reference dataset.

A patient's uploaded photo is reduced to a biomarker vector (plus age) and matched
against reference *Before* images. For each biologic, the expected response is the
distance-weighted average of the improvement scores of the most similar cases.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.schemas.predict import (
    BiologicLikelihood,
    ContributingBiomarker,
    ExactMatch,
    Explanation,
    Heatmap,
    MatchedPatient,
    PatientFeatures,
    PredictResponse,
    SeverityScores,
)
from app.services.biomarker_extraction import extract_biomarkers
from app.services.image_dataset import (
    ImageReferenceCase,
    ImageReferenceRepository,
    image_histogram,
    image_signature,
    signature_similarity,
)

BIOLOGICS = ("Dupixent", "Ebglyss")
REFERENCE_MEDIA_PREFIX = "/api/reference-media"

# Feature weights for patient <-> reference similarity (sum ~= 1.0).
MATCH_WEIGHTS: Dict[str, float] = {
    "inflammation_score": 0.22,
    "erythema_score": 0.20,
    "lesion_coverage_pct": 0.18,
    "texture_score": 0.12,
    "affected_body_area_pct": 0.10,
    "dryness_scaling_score": 0.10,
    "age": 0.08,
}
BIOMARKER_FIELDS = (
    "erythema_score",
    "lesion_coverage_pct",
    "texture_score",
    "dryness_scaling_score",
    "inflammation_score",
    "affected_body_area_pct",
)
BIOMARKER_LABELS: Dict[str, str] = {
    "erythema_score": "redness intensity",
    "lesion_coverage_pct": "lesion coverage",
    "texture_score": "skin texture",
    "dryness_scaling_score": "dryness / scaling",
    "inflammation_score": "overall inflammation",
    "affected_body_area_pct": "affected area",
    "age": "age",
}

TOP_K_NEIGHBORS = 5
MAX_MATCHED_PATIENTS = 3
MAX_REASONS = 3
# Distance below which two features count as an aligned "matching reason".
REASON_DISTANCE_CEILING = 0.25

# --- Discriminative recommendation tuning ---------------------------------
# The winner is decided by a *relative* comparison: for each biomarker, is the
# patient closer to the Dupixent cohort centroid or the Ebglyss one, weighted by
# how much the two cohorts actually differ on that feature. This is balanced by
# construction (symmetric) and driven only by features that genuinely separate the
# groups, instead of absolute nearest-neighbor distance (which favors whichever
# cohort happens to have a case near the query — a dataset artifact that flips with
# the input distribution). The lean is centered on the *typical* reference photo
# (z-scored against the reference cohort) so neither biologic starts ahead.
PREF_Z_FULL_SCALE = 1.0      # z-score at which the display separation saturates
LEAN_SEPARATION = 10.0       # max +/- display points a strong lean adds
BASE_CONF_LOW = 68.0         # display when the photo barely matches any success case
BASE_CONF_SPAN = 22.0        # + this * overall match quality (0..1)
LIKELIHOOD_FLOOR = 40.0      # never show an improbably low number
LIKELIHOOD_CEIL = 99.0       # never read as a literal 100% guarantee

# Perceptual-signature similarity above which an uploaded photo is treated as the
# *same* image as a reference before-photo (our matching-algorithm probe). We take
# the max of a precise structural signature (great for identical files) and a
# crop/translation-tolerant color histogram (great for screenshots/re-crops of a
# reference photo). Different patients top out well below this (structural <=0.60,
# histogram <=0.86), so 0.90 keeps a safe margin against false positives.
EXACT_MATCH_THRESHOLD = 0.90
# Confidence floor applied to the matched biologic when an exact image match is found.
EXACT_MATCH_LIKELIHOOD_FLOOR = 96

DISCLAIMER = (
    "DermaMatch is prototype decision-support for discussion with a dermatologist. "
    "It is not a diagnosis, prescription, or medical advice."
)
PRIVACY_NOTICE = (
    "Uploaded images are analyzed for this session only and are not stored as an "
    "account or EHR record."
)

# Lifestyle / side-effect-profile heuristics derived from the optional "typical day"
# note. These reflect *dosing-convenience and tolerability considerations* a patient
# might raise with their dermatologist — not efficacy claims. Each rule may nudge the
# recommendation toward the biologic whose profile better fits that lifestyle, and/or
# surface a plain-language consideration. Dupixent maintenance is ~every 2 weeks;
# Ebglyss maintenance is ~every 4 weeks after loading.
LIFESTYLE_NUDGE_POINTS = 3
LIFESTYLE_NUDGE_CAP = 6
LIFESTYLE_RULES: Tuple[Tuple[Tuple[str, ...], Optional[str], str], ...] = (
    (
        (
            "travel", "traveling", "travelling", "on the road", "road warrior",
            "flight", "flights", "flying", "fly ", "airport", "airplane", "on planes",
            "frequent flyer", "hotel", "away from home", "trip", "trips",
            "busy", "hectic", "packed schedule", "no time", "always moving",
            "on the go", "on the move", "commute", "commuting", "shift work",
            "long hours", "overtime", "irregular schedule", "unpredictable schedule",
        ),
        "Ebglyss",
        "Your day sounds busy and on-the-go — Ebglyss's less frequent maintenance dosing "
        "(about every 4 weeks vs every 2 weeks) may be easier to keep up with.",
    ),
    (
        (
            "needle", "needles", "injection", "injections", "shot", "shots",
            "afraid of needles", "hate shots", "phobia", "squeamish",
        ),
        "Ebglyss",
        "If frequent injections are a concern, Ebglyss's less frequent maintenance "
        "schedule means fewer shots to manage.",
    ),
    (
        (
            "child", "children", "kids", "toddler", "baby", "parent", "parenting",
            "caregiver", "caring for", "family to look after",
        ),
        "Ebglyss",
        "Caregiving leaves little time for clinic visits — a less frequent dosing "
        "schedule (Ebglyss) may fit a full household routine better.",
    ),
    (
        (
            "outdoor", "outdoors", "outside", "sun", "sweat", "sweating", "gym",
            "run", "running", "jog", "sport", "sports", "athlete", "active",
            "hiking", "hike", "swim", "swimming", "cycling", "bike",
        ),
        None,
        "An active, outdoor routine can affect skin barrier and irritation — worth "
        "raising when you discuss day-to-day tolerability with your dermatologist.",
    ),
)


# Comparative side-effect / tolerability knowledge base (decision-support only, NOT a
# diagnosis or a safety guarantee). Each rule maps the concerns a patient might voice in
# their free-text notes to the biologic that side effect is *more frequently associated
# with* in the AD literature, so we can nudge the estimate toward the better-tolerated
# alternative for that specific concern — and always explain the trade-off in plain
# language. Structure mirrors LIFESTYLE_RULES: the middle value is the biologic to nudge
# TOWARD (the better-tolerated one for that concern), or None when the effect is reported
# with both and we cannot differentiate (surface a monitoring note, no directional nudge).
#
# Directional differentiators used here:
#   • Conjunctivitis / ocular surface disease: reported markedly more often with
#     Dupixent (dupilumab) than Ebglyss (lebrikizumab) -> nudge toward Ebglyss.
#   • Arthralgia / joint pain: reported more often with dupilumab -> nudge toward Ebglyss.
#   • Facial / head-and-neck erythema ("dupilumab facial redness"): a recognized
#     paradoxical dupilumab-associated reaction -> nudge toward Ebglyss.
# Effects reported with both agents (herpes/cold sores, injection-site reactions,
# headache, eosinophilia) are surfaced as monitoring notes without a directional nudge.
SIDE_EFFECT_NUDGE_POINTS = 4
SIDE_EFFECT_NUDGE_CAP = 8
SIDE_EFFECT_RULES: Tuple[Tuple[Tuple[str, ...], Optional[str], str], ...] = (
    (
        (
            "conjunctivitis", "pink eye", "pinkeye", "eye irritation", "eye irritated",
            "irritated eyes", "dry eye", "dry eyes", "eye inflammation", "eye redness",
            "red eyes", "itchy eyes", "eye problems", "eye issues", "blepharitis",
            "keratitis", "contact lens", "contact lenses", "contacts", "watery eyes",
        ),
        "Ebglyss",
        "You raised eye irritation. Conjunctivitis and other eye-surface irritation are "
        "reported notably more often with Dupixent (dupilumab) than with Ebglyss "
        "(lebrikizumab), so we weighted your estimate toward Ebglyss for this concern. "
        "If your recommendation still lands on Dupixent, your skin's biomarker match "
        "outweighed it — not that it was ignored; ask your dermatologist about eye "
        "monitoring and lubricating drops.",
    ),
    (
        (
            "joint pain", "joint aches", "joint ache", "achy joints", "aching joints",
            "sore joints", "joint issues", "joint problems", "my joints", "arthralgia",
            "arthritis", "musculoskeletal", "joint stiffness", "joints",
        ),
        "Ebglyss",
        "You told us you want to avoid joint pain. Joint aches (arthralgia) have been "
        "reported more often with Dupixent (dupilumab) than with Ebglyss (lebrikizumab), "
        "so we nudged your estimate toward Ebglyss. If your recommendation still lands "
        "on Dupixent, your skin's biomarker match was strong enough to outweigh this "
        "concern — not that it was ignored. Either way, flag joint-symptom monitoring "
        "with your dermatologist before starting.",
    ),
    (
        (
            "facial redness", "face redness", "red face", "facial flushing",
            "face flushing", "facial dermatitis", "head and neck dermatitis",
            "red cheeks", "flushed face",
        ),
        "Ebglyss",
        "You mentioned facial redness. A paradoxical head-and-neck / facial erythema is a "
        "recognized reaction associated with Dupixent (dupilumab), so we weighted your "
        "estimate toward Ebglyss (lebrikizumab) for this concern. If Dupixent still comes "
        "out ahead, the biomarker match outweighed it — discuss this specific risk with "
        "your dermatologist.",
    ),
    (
        (
            "cold sore", "cold sores", "oral herpes", "herpes", "fever blister",
            "fever blisters", "hsv", "shingles", "zoster",
        ),
        None,
        "You mentioned herpes / cold sores. Herpes-related infections have been reported "
        "with both Dupixent and Ebglyss, so this does not favor one over the other — but "
        "tell your dermatologist about your history so they can plan monitoring.",
    ),
    (
        (
            "injection site", "injection-site", "injection reaction", "site reaction",
            "redness at the injection", "swelling at the injection", "sore after the shot",
        ),
        None,
        "You mentioned injection-site reactions. Local injection-site reactions are "
        "reported with both biologics, so this concern does not favor one over the other; "
        "your dermatologist can share technique tips to reduce them.",
    ),
    (
        (
            "headache", "headaches", "migraine", "migraines",
        ),
        None,
        "You mentioned headaches. Headache has been reported with both biologics and is "
        "not a differentiator between them — worth noting for your dermatologist.",
    ),
    (
        (
            "eosinophil", "eosinophilia", "high eosinophils",
        ),
        None,
        "You mentioned eosinophils. Transient blood-eosinophil increases can occur with "
        "both biologics; your dermatologist may monitor bloodwork, but it does not favor "
        "one over the other.",
    ),
)


def _match_rule_table(
    text: str,
    rules: Tuple[Tuple[Tuple[str, ...], Optional[str], str], ...],
    points: int,
    cap: int,
) -> Tuple[Dict[str, int], List[str]]:
    """Scan free text against a rule table, accumulating nudges + considerations."""
    nudges: Dict[str, int] = {biologic: 0 for biologic in BIOLOGICS}
    considerations: List[str] = []
    text = (text or "").lower().strip()
    if not text:
        return nudges, considerations
    for keywords, biologic, note in rules:
        if any(keyword in text for keyword in keywords):
            considerations.append(note)
            if biologic in nudges:
                nudges[biologic] = min(cap, nudges[biologic] + points)
    return nudges, considerations


def _analyze_lifestyle(daily_routine: str) -> Tuple[Dict[str, int], List[str]]:
    """Map the free-text 'typical day' to dosing-fit nudges + plain considerations."""
    return _match_rule_table(
        daily_routine, LIFESTYLE_RULES, LIFESTYLE_NUDGE_POINTS, LIFESTYLE_NUDGE_CAP
    )


def _analyze_side_effects(daily_routine: str) -> Tuple[Dict[str, int], List[str]]:
    """Map voiced side-effect concerns to the better-tolerated biologic + explanation."""
    return _match_rule_table(
        daily_routine, SIDE_EFFECT_RULES, SIDE_EFFECT_NUDGE_POINTS, SIDE_EFFECT_NUDGE_CAP
    )


def _apply_lifestyle_nudges(
    likelihoods: List[BiologicLikelihood], nudges: Dict[str, int]
) -> List[BiologicLikelihood]:
    if not any(nudges.values()):
        return likelihoods
    adjusted: List[BiologicLikelihood] = []
    for item in likelihoods:
        delta = nudges.get(item.biologic, 0)
        if delta and item.likelihood_pct > 0:
            new_pct = round(min(95, max(0, item.likelihood_pct + delta)), 1)
            adjusted.append(
                item.model_copy(
                    update={
                        "likelihood_pct": new_pct,
                        "caveat": (
                            f"{item.caveat} Adjusted for lifestyle and comorbidity "
                            f"fit (+{delta})."
                        ),
                    }
                )
            )
        else:
            adjusted.append(item)
    return adjusted


# Dupixent (dupilumab) is FDA-approved not only for atopic dermatitis but also for
# asthma and chronic rhinosinusitis (which underlies allergic-rhinitis / "hay fever"
# type disease). Ebglyss (lebrikizumab) is currently approved for atopic dermatitis
# only. A patient with these atopic comorbidities may therefore favor Dupixent, since
# a single biologic could address more than their skin. This is a treatment-selection
# consideration to raise with a clinician — not an efficacy guarantee.
COMORBIDITY_NUDGE_POINTS = 6
# Only Dupixent (dupilumab) is FDA-approved for the atopic comorbidities we ask about
# (asthma, allergic rhinitis / "hay fever"). Ebglyss (lebrikizumab) is approved for
# atopic dermatitis only. Any comorbidity-approval claim must reference this biologic.
COMORBIDITY_APPROVED_BIOLOGIC = "Dupixent"

_COMORBIDITY_LABELS: Dict[str, str] = {
    "asthma": "asthma",
    "hay-fever": "hay fever (allergic rhinitis)",
    "both": "asthma and hay fever",
}


def _analyze_comorbidities(
    atopic_comorbidities: str,
) -> Tuple[Dict[str, int], List[str]]:
    """Map reported asthma / hay-fever comorbidity to a Dupixent nudge + note."""
    nudges: Dict[str, int] = {biologic: 0 for biologic in BIOLOGICS}
    considerations: List[str] = []
    value = (atopic_comorbidities or "").strip().lower()
    if value not in _COMORBIDITY_LABELS:
        return nudges, considerations
    nudges["Dupixent"] = COMORBIDITY_NUDGE_POINTS
    label = _COMORBIDITY_LABELS[value]
    considerations.append(
        f"You reported {label}. Dupixent is also FDA-approved for these atopic "
        f"conditions, so one biologic could treat more than your skin — worth raising "
        f"with your dermatologist. Ebglyss is currently approved for eczema only."
    )
    return nudges, considerations


# A biologic the patient reports having already tried without success is a strong,
# patient-specific signal to prefer the alternative. It is applied as a sizable display
# penalty (not an absolute veto — a clinician may still weigh a re-trial, dose change,
# or that the failure was for tolerability rather than efficacy) and is ALWAYS surfaced
# in the explanation so the patient never feels their history was ignored.
PRIOR_FAILURE_PENALTY = 20.0
_BIOLOGIC_MENTIONS: Dict[str, Tuple[str, ...]] = {
    "Dupixent": ("dupixent", "dupilumab"),
    "Ebglyss": ("ebglyss", "lebrikizumab"),
}


def _detect_prior_failures(tried_biologics: str, stopped_reason: str) -> List[str]:
    """Which reference biologic(s) the patient reports having tried without success."""
    if (tried_biologics or "").strip().lower() != "yes":
        return []
    text = (stopped_reason or "").lower()
    if not text.strip():
        return []
    return [
        biologic
        for biologic, names in _BIOLOGIC_MENTIONS.items()
        if any(name in text for name in names)
    ]


def _apply_prior_failures(
    likelihoods: List[BiologicLikelihood], prior_failures: List[str]
) -> List[BiologicLikelihood]:
    """Lower the estimate for any biologic the patient already tried unsuccessfully."""
    if not prior_failures:
        return likelihoods
    adjusted: List[BiologicLikelihood] = []
    for item in likelihoods:
        if item.biologic in prior_failures and item.likelihood_pct > 0:
            new_pct = round(
                max(LIKELIHOOD_FLOOR, item.likelihood_pct - PRIOR_FAILURE_PENALTY), 1
            )
            adjusted.append(
                item.model_copy(
                    update={
                        "likelihood_pct": new_pct,
                        "caveat": (
                            f"{item.caveat} You reported already trying {item.biologic} "
                            f"without success, so its estimate is lowered."
                        ),
                    }
                )
            )
        else:
            adjusted.append(item)
    return adjusted


def _merge_nudges(*nudge_maps: Dict[str, int]) -> Dict[str, int]:
    merged: Dict[str, int] = {biologic: 0 for biologic in BIOLOGICS}
    for nudge_map in nudge_maps:
        for biologic, delta in nudge_map.items():
            merged[biologic] = merged.get(biologic, 0) + delta
    return merged


@dataclass(frozen=True)
class _ScoredCase:
    case: ImageReferenceCase
    similarity: float
    distances: Dict[str, float]


def _normalized(features: PatientFeatures, field: str) -> float:
    value = float(getattr(features, field))
    return value / 100.0 if field.endswith("_pct") else value


def _age_bounds(cases: List[ImageReferenceCase]) -> Optional[Tuple[int, int]]:
    ages = [case.age for case in cases if case.age is not None]
    if not ages:
        return None
    return min(ages), max(ages)


def _age_norm(age: int, low: int, high: int) -> float:
    if high <= low:
        return 0.5
    return min(1.0, max(0.0, (age - low) / (high - low)))


def _feature_distances(
    patient: PatientFeatures,
    patient_age_norm: float,
    case: ImageReferenceCase,
    age_bounds: Optional[Tuple[int, int]],
) -> Dict[str, float]:
    distances: Dict[str, float] = {}
    for field in BIOMARKER_FIELDS:
        distances[field] = abs(
            _normalized(patient, field) - _normalized(case.before_features, field)
        )
    # Age only contributes when both the patient and the case have a known age.
    if case.age is not None and age_bounds is not None:
        low, high = age_bounds
        distances["age"] = abs(patient_age_norm - _age_norm(case.age, low, high))
    return distances


def _weighted_similarity(distances: Dict[str, float]) -> float:
    used_weight = sum(MATCH_WEIGHTS[field] for field in distances if field in MATCH_WEIGHTS)
    if used_weight <= 0:
        return 0.0
    total = sum(
        MATCH_WEIGHTS[field] * (distances[field] ** 2)
        for field in distances
        if field in MATCH_WEIGHTS
    )
    distance = math.sqrt(total / used_weight)  # 0..1, normalized over used features
    return float(min(1.0, max(0.0, 1.0 - distance)))


def _score_cases(
    patient: PatientFeatures,
    patient_age: int,
    cases: List[ImageReferenceCase],
) -> List[_ScoredCase]:
    age_bounds = _age_bounds(cases)
    patient_age_norm = (
        _age_norm(patient_age, age_bounds[0], age_bounds[1]) if age_bounds else 0.5
    )
    scored: List[_ScoredCase] = []
    for case in cases:
        distances = _feature_distances(patient, patient_age_norm, case, age_bounds)
        scored.append(
            _ScoredCase(
                case=case,
                similarity=_weighted_similarity(distances),
                distances=distances,
            )
        )
    scored.sort(key=lambda item: item.similarity, reverse=True)
    return scored


def _confidence_label(best_similarity: float, count: int) -> str:
    if count == 0:
        return "no matching cases"
    if best_similarity >= 0.9:
        return "high match confidence"
    if best_similarity >= 0.78:
        return "moderate match confidence"
    return "low match confidence (small reference set)"


def _cohort_centroids(
    cases: List[ImageReferenceCase],
) -> Dict[str, Dict[str, float]]:
    """Mean normalized (0–1) biomarker vector for each biologic's success cases.

    Mean (not median) is deliberate: the high-erythema Dupixent cases lift that
    cohort's redness centroid, which is exactly the "redder skin looks more like a
    Dupixent responder" signal we want to preserve.
    """
    centroids: Dict[str, Dict[str, float]] = {}
    for biologic in BIOLOGICS:
        members = [c for c in cases if c.biologic == biologic]
        centroids[biologic] = {
            field: (
                sum(_normalized(c.before_features, field) for c in members)
                / len(members)
            )
            if members
            else 0.0
            for field in BIOMARKER_FIELDS
        }
    return centroids


def _pref_dupixent(
    patient: PatientFeatures, centroids: Dict[str, Dict[str, float]]
) -> Tuple[float, Dict[str, float]]:
    """Preference for Dupixent vs Ebglyss from a relative, per-feature comparison.

    For each biomarker the patient gets a 0–1 preference (1 = closer to the Dupixent
    centroid, 0 = closer to Ebglyss), weighted by ``MATCH_WEIGHTS[f] * |centroidD −
    centroidE|`` so features that don't separate the cohorts contribute ~nothing.
    Returns ``(pref_dupixent, per-field signed pull toward Dupixent)``.
    """
    dup = centroids["Dupixent"]
    ebg = centroids["Ebglyss"]
    num = den = 0.0
    contrib: Dict[str, float] = {}
    for field in BIOMARKER_FIELDS:
        disc = abs(dup[field] - ebg[field])
        weight = MATCH_WEIGHTS.get(field, 0.0) * disc
        value = _normalized(patient, field)
        dist_d = abs(value - dup[field])
        dist_e = abs(value - ebg[field])
        pref_f = 0.5 if (dist_d + dist_e) < 1e-9 else dist_e / (dist_d + dist_e)
        contrib[field] = weight * (pref_f - 0.5)
        num += weight * pref_f
        den += weight
    pref = num / den if den > 0 else 0.5
    return pref, contrib


def _pref_stats(
    cases: List[ImageReferenceCase], centroids: Dict[str, Dict[str, float]]
) -> Tuple[float, float]:
    """Center (median) and spread (guarded std) of the Dupixent preference.

    Used to z-score a patient's preference so the recommendation leans relative to the
    *typical* reference photo — keeping the split balanced regardless of the absolute
    region the extractor maps real uploads into. The median (not the mean) is the
    center so a right-skewed preference distribution still splits ~50/50.
    """
    prefs = sorted(_pref_dupixent(c.before_features, centroids)[0] for c in cases)
    if not prefs:
        return 0.5, 1.0
    n = len(prefs)
    center = (
        prefs[n // 2] if n % 2 else (prefs[n // 2 - 1] + prefs[n // 2]) / 2.0
    )
    mean = sum(prefs) / n
    var = sum((p - mean) ** 2 for p in prefs) / n
    return center, max(math.sqrt(var), 1e-6)


def _display_likelihoods(
    pref_dupixent: float,
    pref_mean: float,
    pref_std: float,
    overall_match: float,
    scored: List[_ScoredCase],
) -> List[BiologicLikelihood]:
    """Turn the discriminative preference into two friendly, symmetric % displays.

    A shared base confidence (how well the photo matches *any* success case) is split
    by the patient's lean toward one biologic. Recommended always reads higher; a
    neutral photo shows a near-tie; numbers stay in a plausible high range because
    every reference case is a success case. Capped below 100% for medical safety.
    """
    z = (pref_dupixent - pref_mean) / pref_std
    lean = max(-1.0, min(1.0, z / PREF_Z_FULL_SCALE))
    base = BASE_CONF_LOW + BASE_CONF_SPAN * max(0.0, min(1.0, overall_match))
    separation = LEAN_SEPARATION * lean
    raw = {"Dupixent": base + separation, "Ebglyss": base - separation}

    likelihoods: List[BiologicLikelihood] = []
    for biologic in BIOLOGICS:
        matches = [item for item in scored if item.case.biologic == biologic][
            :TOP_K_NEIGHBORS
        ]
        if not matches:
            likelihoods.append(
                BiologicLikelihood(
                    biologic=biologic,
                    likelihood_pct=0,
                    confidence_label="no matching cases",
                    matched_case_count=0,
                    weighted_outcome_score=0.0,
                    caveat=f"No {biologic} success cases in the reference dataset yet.",
                )
            )
            continue
        pct = round(
            max(LIKELIHOOD_FLOOR, min(LIKELIHOOD_CEIL, raw[biologic])), 1
        )
        likelihoods.append(
            BiologicLikelihood(
                biologic=biologic,
                likelihood_pct=pct,
                confidence_label=_confidence_label(overall_match, len(matches)),
                matched_case_count=len(matches),
                weighted_outcome_score=round(pct / 100.0, 3),
                caveat=(
                    f"Based on how closely your skin matches {len(matches)} patient(s) "
                    f"who improved on {biologic}; every reference case is a success case."
                ),
            )
        )
    return likelihoods


def _matching_reasons(scored: _ScoredCase) -> List[str]:
    aligned = sorted(
        (
            (field, distance)
            for field, distance in scored.distances.items()
            if distance <= REASON_DISTANCE_CEILING
        ),
        key=lambda item: item[1],
    )
    reasons = [
        f"Similar {BIOMARKER_LABELS.get(field, field)}" for field, _ in aligned[:MAX_REASONS]
    ]
    if not reasons:
        reasons.append("Closest overall biomarker profile in the dataset")
    return reasons


def _media_url(relative_path: str) -> str:
    return f"{REFERENCE_MEDIA_PREFIX}/{relative_path.lstrip('/')}"


def _matched_patients(scored: List[_ScoredCase]) -> List[MatchedPatient]:
    matched: List[MatchedPatient] = []
    for item in scored[:MAX_MATCHED_PATIENTS]:
        case = item.case
        matched.append(
            MatchedPatient(
                case_id=case.case_id,
                similarity=round(item.similarity, 3),
                biologic_used=case.biologic,
                outcome_label=case.outcome_label,
                outcome_score=round(case.improvement_score, 3),
                demographic_summary=(
                    f"Age {case.age}" if case.age is not None else "Age not recorded"
                ),
                matching_reasons=_matching_reasons(item),
                before_image_url=_media_url(case.before_path),
                after_image_url=_media_url(case.after_path),
            )
        )
    return matched


def _top_biomarkers(patient: PatientFeatures) -> List[ContributingBiomarker]:
    ranked = sorted(
        BIOMARKER_FIELDS,
        key=lambda field: _normalized(patient, field),
        reverse=True,
    )
    contributors: List[ContributingBiomarker] = []
    for field in ranked[:3]:
        contributors.append(
            ContributingBiomarker(
                name=field,
                label=BIOMARKER_LABELS.get(field, field),
                patient_value=round(float(getattr(patient, field)), 3),
                direction="drives the similarity match",
                weight=round(MATCH_WEIGHTS.get(field, 0.0), 2),
            )
        )
    return contributors


def _find_exact_match(
    image_bytes: bytes, cases: List[ImageReferenceCase]
) -> Optional[Tuple[ImageReferenceCase, float]]:
    """Return the reference case whose before-photo is (near-)identical, if any.

    Robust to screenshots/re-crops: combines a precise structural signature with a
    crop/translation-tolerant color histogram and keeps whichever is stronger.
    """
    upload_signature = image_signature(image_bytes)
    upload_histogram = image_histogram(image_bytes)
    best: Optional[Tuple[ImageReferenceCase, float]] = None
    for case in cases:
        structural = signature_similarity(upload_signature, case.before_signature)
        histogram = signature_similarity(upload_histogram, case.before_histogram)
        similarity = max(structural, histogram)
        if similarity >= EXACT_MATCH_THRESHOLD and (best is None or similarity > best[1]):
            best = (case, similarity)
    return best


def _biomarker_differentiators(
    recommended: str, field_contrib: Dict[str, float]
) -> List[Tuple[str, float]]:
    """Rank biomarkers by how much they pull the match toward `recommended`.

    Uses the same per-feature signed pull that decided the recommendation (from
    :func:`_pref_dupixent`), so the explanation always agrees with the pick. A
    positive value means the patient's skin on that feature looks more like the
    recommended cohort; negative means it leans toward the other biologic.
    """
    sign = 1.0 if recommended == "Dupixent" else -1.0
    ranked = [(field, sign * field_contrib.get(field, 0.0)) for field in BIOMARKER_FIELDS]
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked


def _level_word(value: float) -> str:
    if value < 0.2:
        return "very low"
    if value < 0.4:
        return "low"
    if value < 0.6:
        return "moderate"
    if value < 0.8:
        return "high"
    return "very high"


def _recommendation_rationale(
    patient: PatientFeatures,
    best: BiologicLikelihood,
    other: BiologicLikelihood,
    field_contrib: Dict[str, float],
    centroids: Dict[str, Dict[str, float]],
    comorbidity_label: Optional[str],
) -> str:
    diffs = _biomarker_differentiators(best.biologic, field_contrib)
    drivers = [field for field, val in diffs if val > 1e-4]
    against = [field for field, val in diffs if val < -1e-4]
    net = sum(val for _, val in diffs)
    margin = round(best.likelihood_pct - other.likelihood_pct, 1)

    # Comorbidity was the deciding factor: biomarkers alone don't favor the pick.
    if comorbidity_label and net <= 1e-4:
        if best.biologic == COMORBIDITY_APPROVED_BIOLOGIC:
            return (
                f"On skin biomarkers alone, your matches for {best.biologic} and "
                f"{other.biologic} are nearly even. Because you reported {comorbidity_label} "
                f"— which {best.biologic} is also approved to treat — it becomes the better "
                f"overall fit for you."
            )
        return (
            f"On skin biomarkers alone, your matches for {best.biologic} and "
            f"{other.biologic} are nearly even, with your photo leaning slightly toward "
            f"{best.biologic}. Because you reported {comorbidity_label}, it's worth "
            f"discussing {COMORBIDITY_APPROVED_BIOLOGIC} with your dermatologist too — it's "
            f"also FDA-approved for that condition, so one biologic could address more "
            f"than your skin."
        )

    if not drivers:
        return (
            f"{best.biologic} and {other.biologic} are almost tied ("
            f"{best.likelihood_pct:g}% vs {other.likelihood_pct:g}%); your skin resembles "
            f"successful patients from both groups about equally, so this is a close call "
            f"to discuss with your dermatologist."
        )

    rec_means = centroids.get(best.biologic, {})
    oth_means = centroids.get(other.biologic, {})

    def clause(field: str) -> str:
        label = BIOMARKER_LABELS.get(field, field)
        pv = _normalized(patient, field)
        rec = rec_means.get(field, pv)
        oth = oth_means.get(field, pv)
        return (
            f"your {label} is {_level_word(pv)} (~{round(pv * 100)}/100), which lines up "
            f"with the {best.biologic} responders (~{round(rec * 100)}) and sits apart from "
            f"the {other.biologic} responders (~{round(oth * 100)})"
        )

    driver_clauses = "; ".join(clause(field) for field in drivers[:2])
    sentence = f"Specifically, {driver_clauses}."

    if against:
        field = against[0]
        label = BIOMARKER_LABELS.get(field, field)
        pv = _normalized(patient, field)
        oth = oth_means.get(field, pv)
        sentence += (
            f" The one feature pointing the other way is your {label} "
            f"(~{round(pv * 100)}/100 vs the {other.biologic} group's ~{round(oth * 100)})."
        )

    margin_text = f"a {margin:g}-point edge" if margin > 0 else "a razor-thin edge"
    sentence += (
        f" On balance that gives {best.biologic} {margin_text} "
        f"({best.likelihood_pct:g}% vs {other.likelihood_pct:g}%)."
    )
    if comorbidity_label:
        if best.biologic == COMORBIDITY_APPROVED_BIOLOGIC:
            sentence += (
                f" Your reported {comorbidity_label} further supports {best.biologic}, "
                f"which is also FDA-approved to treat it."
            )
        else:
            sentence += (
                f" Separately, because you reported {comorbidity_label}, "
                f"{COMORBIDITY_APPROVED_BIOLOGIC} is also FDA-approved for that condition "
                f"— worth weighing with your dermatologist, even though your skin leans "
                f"{best.biologic}."
            )
    return sentence


def _humanize_list(items: List[str]) -> str:
    """Join names as 'A', 'A and B', or 'A, B, and C'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _explanation(
    patient: PatientFeatures,
    likelihoods: List[BiologicLikelihood],
    total_cases: int,
    exact: Optional[ExactMatch],
    scored: List[_ScoredCase],
    field_contrib: Dict[str, float],
    centroids: Dict[str, Dict[str, float]],
    lifestyle_considerations: Optional[List[str]] = None,
    comorbidity_label: Optional[str] = None,
    prior_failures: Optional[List[str]] = None,
) -> Explanation:
    ranked = sorted(likelihoods, key=lambda item: item.likelihood_pct, reverse=True)
    best = ranked[0]
    top_biomarkers = _top_biomarkers(patient)
    prior_failures = prior_failures or []

    if exact is not None:
        summary = (
            f"Exact image match found: your photo is visually identical to reference "
            f"case {exact.case_id}, who cleared on {exact.biologic}. We are therefore "
            f"highly confident {exact.biologic} is the closest-matched treatment for you."
        )
        rationale = (
            f"{exact.biologic} is recommended because your uploaded photo is a near-exact "
            f"match to reference case {exact.case_id}, a patient who improved on "
            f"{exact.biologic} — the strongest possible image-grounded signal."
        )
    else:
        summary = (
            f"Your photo's visual biomarkers were matched against {total_cases} success "
            f"cases. {best.biologic} is the closest match — your skin most resembles "
            f"patients who improved on {best.biologic}."
        )
        rationale = None
        if len(ranked) > 1:
            rationale = _recommendation_rationale(
                patient, best, ranked[1], field_contrib, centroids, comorbidity_label
            )

    if prior_failures:
        prior_text = _humanize_list(prior_failures)
        if best.biologic in prior_failures:
            # We still land on a biologic the patient told us failed — never silently
            # re-recommend it; explicitly acknowledge their history.
            ack = (
                f" We know you told us {best.biologic} did not work for you before, and we "
                f"did factor that in — but even after lowering its score, your skin's visual "
                f"biomarkers still resemble {best.biologic} responders more closely than the "
                f"alternative. This does not override your experience: please discuss with "
                f"your dermatologist whether a re-trial, a dose or regimen change, or a "
                f"different treatment path makes sense for you."
            )
            summary += ack
            rationale = (rationale + ack) if rationale else ack.strip()
        else:
            # A prior failure pushed us toward the alternative — say so.
            note = (
                f" Because you told us {prior_text} did not work for you, we lowered its "
                f"estimate and focused on {best.biologic} as the better-fitting alternative."
            )
            summary += note
            rationale = (rationale + note) if rationale else note.strip()

    return Explanation(
        summary=summary,
        recommendation_rationale=rationale,
        top_contributing_biomarkers=top_biomarkers,
        lifestyle_considerations=lifestyle_considerations or [],
    )


_IGA_LABELS: Dict[int, str] = {
    0: "Clear",
    1: "Almost clear",
    2: "Mild",
    3: "Moderate",
    4: "Severe",
}


def _easi_area_score(affected_body_area_pct: float) -> int:
    """Map affected body-area percentage to the EASI 0–6 area score bands."""
    pct = max(0.0, min(100.0, affected_body_area_pct))
    if pct <= 0:
        return 0
    if pct < 10:
        return 1
    if pct < 30:
        return 2
    if pct < 50:
        return 3
    if pct < 70:
        return 4
    if pct < 90:
        return 5
    return 6


def _severity_scores(patient: PatientFeatures) -> SeverityScores:
    """Estimate baseline IGA and EASI from the photo's visual biomarkers.

    Prototype proxies only: each EASI sign is scaled 0–3 from a biomarker
    (erythema→redness, induration/papulation→inflammation, excoriation→dryness/
    scaling, lichenification→skin texture), multiplied by the EASI area score
    (0–6) for a 0–72 index. IGA (0–4) is banded from the resulting EASI so the two
    stay consistent. These approximate clinician scoring; they do not replace it.
    """
    erythema = max(0.0, min(1.0, patient.erythema_score))
    induration = max(0.0, min(1.0, patient.inflammation_score))
    excoriation = max(0.0, min(1.0, patient.dryness_scaling_score))
    lichenification = max(0.0, min(1.0, patient.texture_score))

    sign_sum = (erythema + induration + excoriation + lichenification) * 3.0  # 0–12
    area_score = _easi_area_score(patient.affected_body_area_pct)  # 0–6
    easi = round(area_score * sign_sum, 1)  # 0–72

    if easi <= 0:
        iga = 0
    elif easi <= 1.0:
        iga = 1
    elif easi <= 7.0:
        iga = 2
    elif easi <= 21.0:
        iga = 3
    else:
        iga = 4

    return SeverityScores(
        iga=iga,
        iga_label=_IGA_LABELS[iga],
        easi=easi,
        severity_label=_IGA_LABELS[iga],
    )


def build_predict_response(
    image_bytes: bytes,
    age: int,
    repository: ImageReferenceRepository,
    daily_routine: str = "",
    atopic_comorbidities: str = "",
    tried_biologics: str = "",
    biologics_stopped_reason: str = "",
) -> PredictResponse:
    cases = repository.list_cases()
    patient_features, quality = extract_biomarkers(image_bytes)
    scored = _score_cases(patient_features, age, cases)

    centroids = _cohort_centroids(cases)
    pref_mean, pref_std = _pref_stats(cases, centroids)
    pref_dupixent, field_contrib = _pref_dupixent(patient_features, centroids)
    overall_match = scored[0].similarity if scored else 0.0
    likelihoods = _display_likelihoods(
        pref_dupixent, pref_mean, pref_std, overall_match, scored
    )
    matched_patients = _matched_patients(scored)

    lifestyle_nudges, lifestyle_considerations = _analyze_lifestyle(daily_routine)
    side_effect_nudges, side_effect_considerations = _analyze_side_effects(daily_routine)
    comorbidity_nudges, comorbidity_considerations = _analyze_comorbidities(
        atopic_comorbidities
    )
    considerations = (
        comorbidity_considerations
        + side_effect_considerations
        + lifestyle_considerations
    )
    nudges = _merge_nudges(lifestyle_nudges, side_effect_nudges, comorbidity_nudges)
    comorbidity_label = _COMORBIDITY_LABELS.get(
        (atopic_comorbidities or "").strip().lower()
    )
    prior_failures = _detect_prior_failures(tried_biologics, biologics_stopped_reason)

    exact: Optional[ExactMatch] = None
    exact_hit = _find_exact_match(image_bytes, cases)
    if exact_hit is not None:
        case, similarity = exact_hit
        exact = ExactMatch(
            case_id=case.case_id,
            biologic=case.biologic,
            similarity=round(similarity, 4),
            before_image_url=_media_url(case.before_path),
            after_image_url=_media_url(case.after_path),
        )
        # An exact image match is the strongest possible signal: floor the matched
        # biologic to a high-confidence recommendation AND keep the other biologic
        # strictly below it, so the visibly higher number always matches the pick.
        matched_floor = float(EXACT_MATCH_LIKELIHOOD_FLOOR)
        other_ceiling = matched_floor - 2.0
        likelihoods = [
            item.model_copy(
                update={
                    "likelihood_pct": max(item.likelihood_pct, matched_floor),
                    "confidence_label": "very high confidence",
                    "caveat": (
                        f"Your uploaded photo is visually identical to {case.case_id}, "
                        f"who improved on {case.biologic}."
                    ),
                }
            )
            if item.biologic == case.biologic
            else item.model_copy(
                update={
                    "likelihood_pct": min(item.likelihood_pct, other_ceiling),
                }
            )
            for item in likelihoods
        ]
    else:
        # Lifestyle & comorbidity only nudge the recommendation when there is no exact
        # image match (an exact match is a far stronger, image-grounded signal).
        likelihoods = _apply_lifestyle_nudges(likelihoods, nudges)
        # A reported prior treatment failure is patient-grounded reality — apply it
        # last so it can override lifestyle/comorbidity leanings toward that drug.
        likelihoods = _apply_prior_failures(likelihoods, prior_failures)

    warnings: List[str] = list(quality.warnings)
    if len(cases) < 10:
        warnings.append("small_reference_dataset")

    return PredictResponse(
        request_id=f"dm-{uuid.uuid4().hex[:8]}",
        mock=False,
        disclaimer=DISCLAIMER,
        privacy_notice=PRIVACY_NOTICE,
        patient_features=patient_features,
        severity=_severity_scores(patient_features),
        likelihoods=likelihoods,
        explanation=_explanation(
            patient_features,
            likelihoods,
            len(cases),
            exact,
            scored,
            field_contrib,
            centroids,
            lifestyle_considerations=considerations,
            comorbidity_label=comorbidity_label,
            prior_failures=prior_failures,
        ),
        heatmap=Heatmap(
            overlay_url=None,
            legend="Visual biomarker overlay is not yet rendered in this prototype.",
        ),
        matched_patients=matched_patients,
        warnings=warnings,
        exact_match=exact,
    )
