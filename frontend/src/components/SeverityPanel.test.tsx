import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { SeverityPanel } from './SeverityPanel';
import type { SeverityScores } from '../api/dermaMatchClient';

const severity: SeverityScores = {
  iga: 3,
  iga_label: 'Moderate',
  easi: 18.4,
  easi_max: 72.0,
  severity_label: 'Moderate',
};

describe('SeverityPanel', () => {
  it('renders IGA and EASI values from the API response', () => {
    render(<SeverityPanel severity={severity} />);

    expect(screen.getByRole('heading', { name: /estimated baseline severity/i })).toBeInTheDocument();
    expect(screen.getByText('IGA')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
    expect(screen.getByText('/4')).toBeInTheDocument();
    expect(screen.getByText('Moderate')).toBeInTheDocument();

    expect(screen.getByText('EASI')).toBeInTheDocument();
    expect(screen.getByText('18.4')).toBeInTheDocument();
    expect(screen.getByText('/72')).toBeInTheDocument();
  });
});
