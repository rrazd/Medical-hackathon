"""Tests for lifestyle / side-effect concern classification (LLM + keyword fallback)."""

from unittest.mock import MagicMock, patch

from app.config import Settings
from app.services import concerns


def _fake_openai_response(concern_ids):
    resp = MagicMock()
    resp.raise_for_status = lambda: None
    ids_json = "[" + ", ".join(f'"{cid}"' for cid in concern_ids) + "]"
    resp.json = lambda: {
        "choices": [{"message": {"content": '{"concerns": ' + ids_json + "}"}}]
    }
    return resp


def test_keyword_fallback_when_no_api_key():
    settings = Settings(openai_api_key="")
    ids = concerns.classify_concerns("I fly every week and have joint pain", settings)
    assert "busy_travel" in ids
    assert "joint_pain" in ids


def test_empty_text_returns_no_concerns():
    assert concerns.classify_concerns("   ", Settings(openai_api_key="")) == []


def test_llm_path_catches_phrasing_keywords_miss():
    # "scratchy, gritty eyes" is not in the keyword list, but the LLM can classify it.
    text = "my eyes feel scratchy and gritty all the time"
    assert "eye_irritation" not in concerns._match_keywords(text)

    settings = Settings(openai_api_key="sk-test")
    with patch("httpx.post", return_value=_fake_openai_response(["eye_irritation"])):
        ids = concerns.classify_concerns(text, settings)
    assert ids == ["eye_irritation"]


def test_llm_ignores_unknown_ids_and_preserves_registry_order():
    settings = Settings(openai_api_key="sk-test")
    # Model returns a bogus id + out-of-order valid ids.
    with patch(
        "httpx.post",
        return_value=_fake_openai_response(["joint_pain", "not_a_real_id", "busy_travel"]),
    ):
        ids = concerns.classify_concerns("whatever", settings)
    assert "not_a_real_id" not in ids
    # Registry order: busy_travel is defined before joint_pain.
    assert ids == ["busy_travel", "joint_pain"]


def test_llm_failure_falls_back_to_keywords():
    settings = Settings(openai_api_key="sk-test")
    with patch("httpx.post", side_effect=RuntimeError("boom")):
        ids = concerns.classify_concerns("I have joint pain", settings)
    assert ids == ["joint_pain"]


def test_concerns_to_signals_nudges_and_caps():
    # Two side-effect concerns toward Ebglyss: 4 + 4 = 8 (within cap 8).
    nudges, se_notes, ls_notes = concerns.concerns_to_signals(
        ["eye_irritation", "joint_pain"]
    )
    assert nudges["Ebglyss"] == 8
    assert nudges["Dupixent"] == 0
    assert len(se_notes) == 2
    assert len(ls_notes) == 0


def test_concerns_to_signals_non_directional_adds_note_without_nudge():
    nudges, se_notes, ls_notes = concerns.concerns_to_signals(["herpes"])
    assert nudges["Ebglyss"] == 0
    assert nudges["Dupixent"] == 0
    assert len(se_notes) == 1
