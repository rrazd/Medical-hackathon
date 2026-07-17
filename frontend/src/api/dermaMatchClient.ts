import type { IntakeFormValues } from '../types/intake';

export type BiologicLikelihood = {
  biologic: 'Dupixent' | 'Ebglyss' | string;
  likelihood_pct: number;
  confidence_label: string;
  matched_case_count: number;
  weighted_outcome_score: number;
  caveat: string;
};

export type PatientFeatures = {
  erythema_score: number;
  lesion_coverage_pct: number;
  texture_score: number;
  dryness_scaling_score: number;
  inflammation_score: number;
  affected_body_area_pct: number;
};

export type ContributingBiomarker = {
  name: string;
  label: string;
  patient_value: number;
  direction: string;
  weight: number;
};

export type Explanation = {
  summary: string;
  top_contributing_biomarkers: ContributingBiomarker[];
  lifestyle_considerations?: string[];
};

export type Heatmap = {
  overlay_url: string | null;
  legend: string;
};

export type MatchedPatient = {
  case_id: string;
  similarity: number;
  biologic_used: string;
  outcome_label: string;
  outcome_score: number;
  demographic_summary: string;
  matching_reasons: string[];
  before_image_url: string | null;
  after_image_url: string | null;
};

export type ExactMatch = {
  case_id: string;
  biologic: string;
  similarity: number;
  before_image_url: string | null;
  after_image_url: string | null;
};

export type PredictResponse = {
  request_id: string;
  mock: boolean;
  disclaimer: string;
  privacy_notice: string;
  patient_features: PatientFeatures;
  likelihoods: BiologicLikelihood[];
  explanation: Explanation;
  heatmap: Heatmap;
  matched_patients: MatchedPatient[];
  warnings: string[];
  exact_match?: ExactMatch | null;
};

export async function predict(values: IntakeFormValues, image: File): Promise<PredictResponse> {
  const body = new FormData();
  body.append('image', image);
  body.append('age', String(values.age));
  body.append('sex', values.sex);
  body.append('race_ethnicity', values.race_ethnicity);
  body.append('body_area', values.body_area);
  body.append('prior_treatments', values.prior_treatments ?? '');
  body.append('daily_routine', values.daily_routine ?? '');

  const response = await fetch('/api/predict', { method: 'POST', body });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      if (data && typeof data.detail === 'string') detail = data.detail;
    } catch {
      /* fall back to status text */
    }
    throw new Error(detail);
  }
  return response.json() as Promise<PredictResponse>;
}
