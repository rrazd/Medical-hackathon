import { render, screen, within } from '@testing-library/react';
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
  it('renders the explanation summary and driver rows from the response', () => {
    render(<ExplanationPanel explanation={explanation} />);

    expect(screen.getByText(/this result shell will later describe visual biomarkers/i)).toBeInTheDocument();

    const drivers = screen.getByRole('list', { name: /top contributing biomarker drivers/i });
    expect(within(drivers).getByRole('heading', { name: 'lesion coverage' })).toBeInTheDocument();
    expect(within(drivers).getByText('lesion_coverage_pct')).toBeInTheDocument();
    expect(within(drivers).getByText('28.4')).toBeInTheDocument();
    expect(within(drivers).getByText('similar to responders')).toBeInTheDocument();
    expect(within(drivers).getByText('0.30')).toBeInTheDocument();
    expect(within(drivers).getByRole('heading', { name: 'redness intensity' })).toBeInTheDocument();
    expect(within(drivers).getByText('0.25')).toBeInTheDocument();
  });
});
