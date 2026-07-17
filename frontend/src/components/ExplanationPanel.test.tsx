import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { ExplanationPanel } from './ExplanationPanel';
import type { Explanation } from '../api/dermaMatchClient';

const explanation: Explanation = {
  summary: 'Mock explanation: this result shell will later describe visual biomarkers and similar reference cases.',
  top_contributing_biomarkers: [
    {
      name: 'lesion_coverage_pct',
      label: 'lesion coverage',
      patient_value: 28.4,
      direction: 'similar to responders',
      weight: 0.3,
    },
    {
      name: 'erythema_score',
      label: 'redness intensity',
      patient_value: 0.62,
      direction: 'similar to partial responders',
      weight: 0.25,
    },
  ],
};

describe('ExplanationPanel', () => {
  it('renders the explanation summary and the curated-database biomarker sentence', () => {
    render(<ExplanationPanel explanation={explanation} />);

    expect(screen.getByText(/this result shell will later describe visual biomarkers/i)).toBeInTheDocument();
    expect(
      screen.getByText(/lesion coverage, affected area,\s*overall inflammation/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/curated database of real before\/after cases/i)).toBeInTheDocument();
  });
});
