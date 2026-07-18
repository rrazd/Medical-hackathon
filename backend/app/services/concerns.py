"""Lifestyle & side-effect concern detection for treatment matching.

A patient describes their day-to-day life and worries in free text. We map that to a
small, fixed set of *curated* concern categories — each with vetted, medically-framed
note text and (optionally) a nudge toward the better-tolerated biologic. Two detection
backends produce the same category ids:

  1. LLM classifier (when an OpenAI key is configured): robust to arbitrary phrasing.
     The model ONLY selects from the fixed category ids below — it never authors medical
     claims. All patient-facing wording comes from our curated `note` text.
  2. Keyword fallback (default / offline): deterministic substring matching.

Downstream code turns the detected ids into likelihood nudges + explanation notes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Nudge magnitudes per group. Lifestyle/dosing-fit is a softer signal than a specific
# side-effect the patient wants to avoid. Each group caps its own contribution per
# biologic; the two groups can still combine.
LIFESTYLE_NUDGE_POINTS = 3
LIFESTYLE_NUDGE_CAP = 6
SIDE_EFFECT_NUDGE_POINTS = 4
SIDE_EFFECT_NUDGE_CAP = 8


@dataclass(frozen=True)
class Concern:
    id: str
    group: str  # "lifestyle" | "side_effect"
    nudge_biologic: Optional[str]  # biologic to nudge TOWARD, or None (no directional bias)
    description: str  # shown to the LLM so it can decide if the category applies
    keywords: Tuple[str, ...]  # deterministic fallback matching
    note: str  # curated, patient-facing explanation


# fmt: off
CONCERNS: Tuple[Concern, ...] = (
    # ---- Lifestyle / dosing-fit (nudge toward the more convenient maintenance schedule) ----
    Concern(
        id="busy_travel",
        group="lifestyle",
        nudge_biologic="Ebglyss",
        description=(
            "The patient has a busy, on-the-go, or travel-heavy life: frequent travel, "
            "flights, business trips, long/irregular hours, shift work, commuting, or a "
            "generally hectic unpredictable schedule that makes frequent dosing hard."
        ),
        keywords=(
            "travel", "traveling", "travelling", "on the road", "road warrior",
            "flight", "flights", "flying", "fly ", "airport", "airplane", "on planes",
            "frequent flyer", "hotel", "away from home", "trip", "trips",
            "busy", "hectic", "packed schedule", "no time", "always moving",
            "on the go", "on the move", "commute", "commuting", "shift work",
            "long hours", "overtime", "irregular schedule", "unpredictable schedule",
        ),
        note=(
            "Your day sounds busy and on-the-go — Ebglyss's less frequent maintenance "
            "dosing (about every 4 weeks vs every 2 weeks) may be easier to keep up with."
        ),
    ),
    Concern(
        id="needle_averse",
        group="lifestyle",
        nudge_biologic="Ebglyss",
        description=(
            "The patient dislikes, fears, or wants to minimize injections/needles/shots."
        ),
        keywords=(
            "needle", "needles", "injection", "injections", "shot", "shots",
            "afraid of needles", "hate shots", "phobia", "squeamish",
        ),
        note=(
            "If frequent injections are a concern, Ebglyss's less frequent maintenance "
            "schedule means fewer shots to manage."
        ),
    ),
    Concern(
        id="caregiver",
        group="lifestyle",
        nudge_biologic="Ebglyss",
        description=(
            "The patient is a busy parent or caregiver with little time for clinic visits."
        ),
        keywords=(
            "child", "children", "kids", "toddler", "baby", "parent", "parenting",
            "caregiver", "caring for", "family to look after",
        ),
        note=(
            "Caregiving leaves little time for clinic visits — a less frequent dosing "
            "schedule (Ebglyss) may fit a full household routine better."
        ),
    ),
    Concern(
        id="active_outdoor",
        group="lifestyle",
        nudge_biologic=None,
        description=(
            "The patient has an active, athletic, sweaty, or outdoor/sun-exposed routine "
            "(gym, running, sports, hiking, swimming, cycling, lots of time outside)."
        ),
        keywords=(
            "outdoor", "outdoors", "outside", "sun", "sweat", "sweating", "gym",
            "run", "running", "jog", "sport", "sports", "athlete", "active",
            "hiking", "hike", "swim", "swimming", "cycling", "bike", "workout",
            "work out", "exercise", "lifting", "weights", "crossfit", "yoga",
        ),
        note=(
            "An active, outdoor routine can affect skin barrier and irritation — worth "
            "raising when you discuss day-to-day tolerability with your dermatologist."
        ),
    ),
    # ---- Side-effect avoidance ----
    Concern(
        id="eye_irritation",
        group="side_effect",
        nudge_biologic="Ebglyss",
        description=(
            "The patient wants to avoid, or is prone to, eye problems: conjunctivitis, "
            "pink eye, dry/irritated/red/itchy eyes, blepharitis, keratitis, or wears "
            "contact lenses and is worried about eye side effects."
        ),
        keywords=(
            "conjunctivitis", "pink eye", "pinkeye", "eye irritation", "eye irritated",
            "irritated eyes", "dry eye", "dry eyes", "eye inflammation", "eye redness",
            "red eyes", "itchy eyes", "eye problems", "eye issues", "blepharitis",
            "keratitis", "contact lens", "contact lenses", "contacts", "watery eyes",
        ),
        note=(
            "You raised eye irritation. Conjunctivitis and other eye-surface irritation "
            "are reported notably more often with Dupixent (dupilumab) than with Ebglyss "
            "(lebrikizumab), so we weighted your estimate toward Ebglyss for this concern. "
            "If your recommendation still lands on Dupixent, your skin's biomarker match "
            "outweighed it — not that it was ignored; ask your dermatologist about eye "
            "monitoring and lubricating drops."
        ),
    ),
    Concern(
        id="joint_pain",
        group="side_effect",
        nudge_biologic="Ebglyss",
        description=(
            "The patient wants to avoid, or already has, joint pain / arthralgia / "
            "arthritis / achy or stiff joints / musculoskeletal pain."
        ),
        keywords=(
            "joint pain", "joint aches", "joint ache", "achy joints", "aching joints",
            "sore joints", "joint issues", "joint problems", "my joints", "arthralgia",
            "arthritis", "musculoskeletal", "joint stiffness", "joints",
        ),
        note=(
            "You told us you want to avoid joint pain. Joint aches (arthralgia) have been "
            "reported more often with Dupixent (dupilumab) than with Ebglyss "
            "(lebrikizumab), so we nudged your estimate toward Ebglyss. If your "
            "recommendation still lands on Dupixent, your skin's biomarker match was "
            "strong enough to outweigh this concern — not that it was ignored. Either "
            "way, flag joint-symptom monitoring with your dermatologist before starting."
        ),
    ),
    Concern(
        id="facial_redness",
        group="side_effect",
        nudge_biologic="Ebglyss",
        description=(
            "The patient is worried about facial or head-and-neck redness/flushing "
            "(distinct from their eczema) as a treatment side effect."
        ),
        keywords=(
            "facial redness", "face redness", "red face", "facial flushing",
            "face flushing", "facial dermatitis", "head and neck dermatitis",
            "red cheeks", "flushed face",
        ),
        note=(
            "You mentioned facial redness. A paradoxical head-and-neck / facial erythema "
            "is a recognized reaction associated with Dupixent (dupilumab), so we weighted "
            "your estimate toward Ebglyss (lebrikizumab) for this concern. If Dupixent "
            "still comes out ahead, the biomarker match outweighed it — discuss this "
            "specific risk with your dermatologist."
        ),
    ),
    Concern(
        id="herpes",
        group="side_effect",
        nudge_biologic=None,
        description=(
            "The patient mentions cold sores, oral herpes (HSV), or shingles (zoster)."
        ),
        keywords=(
            "cold sore", "cold sores", "oral herpes", "herpes", "fever blister",
            "fever blisters", "hsv", "shingles", "zoster",
        ),
        note=(
            "You mentioned herpes / cold sores. Herpes-related infections have been "
            "reported with both Dupixent and Ebglyss, so this does not favor one over the "
            "other — but tell your dermatologist about your history so they can plan "
            "monitoring."
        ),
    ),
    Concern(
        id="injection_site",
        group="side_effect",
        nudge_biologic=None,
        description=(
            "The patient is worried about injection-site reactions (redness, swelling, "
            "or soreness where the shot is given)."
        ),
        keywords=(
            "injection site", "injection-site", "injection reaction", "site reaction",
            "redness at the injection", "swelling at the injection", "sore after the shot",
        ),
        note=(
            "You mentioned injection-site reactions. Local injection-site reactions are "
            "reported with both biologics, so this concern does not favor one over the "
            "other; your dermatologist can share technique tips to reduce them."
        ),
    ),
    Concern(
        id="headache",
        group="side_effect",
        nudge_biologic=None,
        description="The patient is worried about headaches or migraines.",
        keywords=("headache", "headaches", "migraine", "migraines"),
        note=(
            "You mentioned headaches. Headache has been reported with both biologics and "
            "is not a differentiator between them — worth noting for your dermatologist."
        ),
    ),
    Concern(
        id="eosinophilia",
        group="side_effect",
        nudge_biologic=None,
        description="The patient mentions eosinophils / eosinophilia / high eosinophil counts.",
        keywords=("eosinophil", "eosinophilia", "high eosinophils"),
        note=(
            "You mentioned eosinophils. Transient blood-eosinophil increases can occur "
            "with both biologics; your dermatologist may monitor bloodwork, but it does "
            "not favor one over the other."
        ),
    ),
)
# fmt: on

_BY_ID: Dict[str, Concern] = {c.id: c for c in CONCERNS}
_VALID_IDS = set(_BY_ID)


def _match_keywords(text: str) -> List[str]:
    """Deterministic fallback: which concern ids appear as substrings in the text."""
    lowered = (text or "").lower()
    if not lowered.strip():
        return []
    return [c.id for c in CONCERNS if any(k in lowered for k in c.keywords)]


def _build_llm_messages(text: str) -> List[Dict[str, str]]:
    catalog = "\n".join(f"- {c.id}: {c.description}" for c in CONCERNS)
    system = (
        "You classify a patient's free-text description of their lifestyle and health "
        "concerns for an atopic-dermatitis biologic treatment tool. From the fixed "
        "category list, return ONLY the categories the patient clearly expresses. Do not "
        "infer medical conditions or side effects they did not mention. Do not invent "
        "categories. Respond as JSON: {\"concerns\": [\"<id>\", ...]}. Use an empty list "
        "if none apply.\n\nCategories:\n" + catalog
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text.strip()},
    ]


def _classify_with_llm(text: str, settings: Settings) -> Optional[List[str]]:
    """Call OpenAI to classify concerns. Returns None on any failure (-> fallback)."""
    try:
        import httpx

        resp = httpx.post(
            f"{settings.openai_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.openai_model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": _build_llm_messages(text),
            },
            timeout=settings.llm_timeout_seconds,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        raw = parsed.get("concerns", []) if isinstance(parsed, dict) else []
        # Keep only known ids, preserve registry order, de-duplicate.
        selected = {cid for cid in raw if cid in _VALID_IDS}
        return [c.id for c in CONCERNS if c.id in selected]
    except Exception as exc:  # noqa: BLE001 - never let classification break a prediction
        logger.warning("LLM concern classification failed, using keyword fallback: %s", exc)
        return None


def classify_concerns(text: str, settings: Optional[Settings] = None) -> List[str]:
    """Return the concern ids expressed in the patient's free text.

    Uses the LLM classifier when an OpenAI key is configured; otherwise (or on any LLM
    error) falls back to deterministic keyword matching.
    """
    if not (text or "").strip():
        return []
    settings = settings or get_settings()
    if settings.openai_api_key.strip():
        ids = _classify_with_llm(text, settings)
        if ids is not None:
            return ids
    return _match_keywords(text)


def concerns_to_signals(
    concern_ids: List[str],
) -> Tuple[Dict[str, int], List[str], List[str]]:
    """Turn detected concern ids into (nudges, side_effect_notes, lifestyle_notes).

    Nudges accumulate per biologic within each group, each group capped independently.
    """
    from app.services.image_predict import BIOLOGICS  # local import to avoid a cycle

    nudges: Dict[str, int] = {biologic: 0 for biologic in BIOLOGICS}
    group_totals: Dict[str, Dict[str, int]] = {
        "lifestyle": {b: 0 for b in BIOLOGICS},
        "side_effect": {b: 0 for b in BIOLOGICS},
    }
    side_effect_notes: List[str] = []
    lifestyle_notes: List[str] = []

    for concern in CONCERNS:  # registry order = stable, sensible display order
        if concern.id not in concern_ids:
            continue
        if concern.group == "side_effect":
            side_effect_notes.append(concern.note)
            points, cap = SIDE_EFFECT_NUDGE_POINTS, SIDE_EFFECT_NUDGE_CAP
        else:
            lifestyle_notes.append(concern.note)
            points, cap = LIFESTYLE_NUDGE_POINTS, LIFESTYLE_NUDGE_CAP
        if concern.nudge_biologic in nudges:
            group_totals[concern.group][concern.nudge_biologic] = min(
                cap, group_totals[concern.group][concern.nudge_biologic] + points
            )

    for group in group_totals.values():
        for biologic, delta in group.items():
            nudges[biologic] += delta

    return nudges, side_effect_notes, lifestyle_notes
