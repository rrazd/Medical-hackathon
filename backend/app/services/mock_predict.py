from app.schemas.predict import (
    BiologicLikelihood,
    ContributingBiomarker,
    Explanation,
    Heatmap,
    MatchedPatient,
    PatientFeatures,
    PredictResponse,
    SeverityScores,
)


def build_mock_predict_response() -> PredictResponse:
    features = PatientFeatures(
        erythema_score=0.62,
        lesion_coverage_pct=28.4,
        texture_score=0.47,
        dryness_scaling_score=0.55,
        inflammation_score=0.68,
        affected_body_area_pct=12.5,
    )

    return PredictResponse(
        request_id="mock-001",
        mock=True,
        disclaimer=(
            "DermaMatch is prototype decision-support for discussion with a dermatologist. "
            "It is not a diagnosis, prescription, or medical advice."
        ),
        privacy_notice=(
            "Uploaded images are analyzed for this session only and are not stored as an "
            "account or EHR record."
        ),
        patient_features=features,
        severity=SeverityScores(
            iga=3,
            iga_label="Moderate",
            easi=18.0,
            severity_label="Moderate",
        ),
        likelihoods=[
            BiologicLikelihood(
                biologic="Dupixent",
                likelihood_pct=72,
                confidence_label="mock prototype estimate",
                matched_case_count=5,
                weighted_outcome_score=0.72,
                caveat="Mock value; real matching comes in a later phase.",
            ),
            BiologicLikelihood(
                biologic="Ebglyss",
                likelihood_pct=64,
                confidence_label="mock prototype estimate",
                matched_case_count=4,
                weighted_outcome_score=0.64,
                caveat="Mock value; real matching comes in a later phase.",
            ),
        ],
        explanation=Explanation(
            summary=(
                "Mock explanation: this result shell will later describe visual biomarkers "
                "and similar reference cases."
            ),
            top_contributing_biomarkers=[
                ContributingBiomarker(
                    name="lesion_coverage_pct",
                    label="lesion coverage",
                    patient_value=features.lesion_coverage_pct,
                    direction="similar to responders",
                    weight=0.30,
                ),
                ContributingBiomarker(
                    name="erythema_score",
                    label="redness intensity",
                    patient_value=features.erythema_score,
                    direction="similar to partial responders",
                    weight=0.25,
                ),
            ],
        ),
        heatmap=Heatmap(
            overlay_url=None,
            legend="Heatmap placeholder; real visual biomarker overlay comes later.",
        ),
        matched_patients=[
            MatchedPatient(
                case_id="MOCK-001",
                similarity=0.91,
                biologic_used="Dupixent",
                outcome_label="responder",
                outcome_score=0.85,
                demographic_summary="Mock case: adult with arm involvement",
                matching_reasons=["similar lesion coverage", "same body area"],
                before_image_url=None,
                after_image_url=None,
            )
        ],
        warnings=["Mock response only; no clinical inference has been performed."],
    )
