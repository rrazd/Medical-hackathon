import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { ResultCards } from './ResultCards';
import type { BiologicLikelihood } from '../api/dermaMatchClient';

const likelihoods: BiologicLikelihood[] = [
  {
    biologic: 'Dupixent',
    likelihood_pct: 72,
    confidence_label: 'mock prototype estimate',
    matched_case_count: 5,
    weighted_outcome_score: 0.72,
    caveat: 'Mock value; real matching comes in a later phase.',
  },
  {
    biologic: 'Ebglyss',
    likelihood_pct: 64,
    confidence_label: 'mock prototype estimate',
    matched_case_count: 4,
    weighted_outcome_score: 0.64,
    caveat: 'Mock value; real matching comes in a later phase.',
  },
];

describe('ResultCards', () => {
  it('renders per-biologic likelihood details from the API response', () => {
    render(<ResultCards likelihoods={likelihoods} />);

    expect(screen.getByRole('heading', { name: /dupixent vs ebglyss likelihood comparison/i })).toBeInTheDocument();

    const dupixent = screen.getByRole('heading', { name: 'Dupixent' }).closest('article');
    const ebglyss = screen.getByRole('heading', { name: 'Ebglyss' }).closest('article');

    expect(dupixent).not.toBeNull();
    expect(ebglyss).not.toBeNull();
    expect(within(dupixent as HTMLElement).getByRole('img', { name: /72 % likelihood/i })).toBeInTheDocument();
    expect(within(dupixent as HTMLElement).getByText('5 mock cases')).toBeInTheDocument();
    expect(within(dupixent as HTMLElement).getByText('0.72')).toBeInTheDocument();
    expect(within(ebglyss as HTMLElement).getByRole('img', { name: /64 % likelihood/i })).toBeInTheDocument();
    expect(within(ebglyss as HTMLElement).getByText('4 mock cases')).toBeInTheDocument();
    expect(within(ebglyss as HTMLElement).getByText('0.64')).toBeInTheDocument();
    expect(screen.getAllByText('Mock value; real matching comes in a later phase.')).toHaveLength(2);
  });
});
