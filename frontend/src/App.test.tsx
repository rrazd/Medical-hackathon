import { cleanup, render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import App from './App';

function renderApp() {
  const queryClient = new QueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>,
  );
}

const mockPredictResponse = {
  request_id: 'mock-001',
  mock: true,
  disclaimer:
    'DermaMatch is prototype decision-support for discussion with a dermatologist. It is not a diagnosis, prescription, or medical advice.',
  privacy_notice: 'Uploaded images are analyzed for this session only and are not stored as an account or EHR record.',
  patient_features: {
    erythema_score: 0.62,
    lesion_coverage_pct: 28.4,
    texture_score: 0.47,
    dryness_scaling_score: 0.55,
    inflammation_score: 0.68,
    affected_body_area_pct: 12.5,
  },
  likelihoods: [
    {
      biologic: 'Dupixent',
      likelihood_pct: 72,
      confidence_label: 'mock prototype estimate',
      matched_case_count: 5,
      weighted_outcome_score: 0.72,
      caveat: 'Mock value.',
    },
    {
      biologic: 'Ebglyss',
      likelihood_pct: 64,
      confidence_label: 'mock prototype estimate',
      matched_case_count: 4,
      weighted_outcome_score: 0.64,
      caveat: 'Mock value.',
    },
  ],
  explanation: {
    summary: 'Mock explanation: this result shell will later describe visual biomarkers and similar reference cases.',
    top_contributing_biomarkers: [
      {
        name: 'lesion_coverage_pct',
        label: 'lesion coverage',
        patient_value: 28.4,
        direction: 'similar to responders',
        weight: 0.3,
      },
    ],
  },
  heatmap: { overlay_url: null, legend: 'Heatmap placeholder; real visual biomarker overlay comes later.' },
  matched_patients: [
    {
      case_id: 'MOCK-001',
      similarity: 0.91,
      biologic_used: 'Dupixent',
      outcome_label: 'responder',
      outcome_score: 0.85,
      demographic_summary: 'Mock case: adult with arm involvement',
      matching_reasons: ['similar lesion coverage', 'same body area'],
      before_image_url: null,
      after_image_url: null,
    },
  ],
  warnings: ['Mock response only.'],
};

describe('App', () => {
  beforeEach(() => {
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      writable: true,
      value: vi.fn(() => 'blob:baseline-preview'),
    });
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      writable: true,
      value: vi.fn(),
    });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('renders safety and privacy framing and a CTA on the landing page', () => {
    renderApp();

    expect(screen.getByText(/not a diagnosis, prescription, or medical advice/i)).toBeInTheDocument();
    expect(screen.getByText(/not stored as an account or EHR record/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /get started/i })).toBeInTheDocument();
  });

  it('blocks incomplete intake and serializes a valid image plus intake to the API contract', async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn<typeof fetch>(
      async (_input, _init) =>
        new Response(JSON.stringify(mockPredictResponse), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
    );
    vi.stubGlobal('fetch', fetchMock);

    renderApp();

    // Enter the wizard.
    await user.click(screen.getByRole('button', { name: /get started/i }));

    // Step 1 — upload a photo, then advance.
    const image = new File(['image'], 'baseline.png', { type: 'image/png' });
    await user.upload(screen.getByLabelText(/baseline AD photo/i), image);
    await user.click(screen.getByRole('button', { name: /next/i }));

    // Step 2 — advancing with empty intake is blocked and does not call the API.
    await user.click(screen.getByRole('button', { name: /next/i }));
    expect(await screen.findByText(/age is required/i)).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();

    await user.type(screen.getByRole('spinbutton', { name: /age/i }), '36');
    await user.selectOptions(screen.getByRole('combobox', { name: /sex/i }), 'female');
    await user.type(screen.getByLabelText(/race\/ethnicity/i), 'Latina');
    await user.selectOptions(screen.getByRole('combobox', { name: /fitzpatrick skin type/i }), 'IV');
    await user.type(screen.getByLabelText(/body area/i), 'forearms');
    await user.type(screen.getByLabelText(/prior treatments/i), 'topical steroids');
    await user.selectOptions(screen.getByRole('combobox', { name: /baseline severity/i }), 'moderate');
    await user.click(screen.getByRole('button', { name: /next/i }));

    // Step 3 — review, then run the estimate.
    await user.click(screen.getByRole('button', { name: /estimate response/i }));

    expect(await screen.findByRole('heading', { name: /your treatment response estimate/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /dupixent vs ebglyss likelihood comparison/i })).toBeInTheDocument();
    expect(screen.getByText(/not a diagnosis, prescription, or medical advice/i)).toBeInTheDocument();
    expect(screen.getByText(/not stored as an account or EHR record/i)).toBeInTheDocument();
    expect(screen.getByText('0.72')).toBeInTheDocument();
    expect(screen.queryByText(/biomarker heatmap placeholder/i)).not.toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /patients like you/i })).toBeInTheDocument();
    const formData = fetchMock.mock.calls[0][1]?.body as FormData;
    expect(formData.get('image')).toBe(image);
    expect(formData.get('age')).toBe('36');
    expect(formData.get('sex')).toBe('female');
    expect(formData.get('race_ethnicity')).toBe('Latina');
    expect(formData.get('fitzpatrick_skin_type')).toBe('IV');
    expect(formData.get('body_area')).toBe('forearms');
    expect(formData.get('prior_treatments')).toBe('topical steroids');
    expect(formData.get('baseline_severity')).toBe('moderate');
  });
});
