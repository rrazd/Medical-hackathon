import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';

import { predict, type PredictResponse } from './api/dermaMatchClient';
import { ExactMatchBanner } from './components/ExactMatchBanner';
import { ExplanationPanel } from './components/ExplanationPanel';
import { HeatmapPlaceholder } from './components/HeatmapPlaceholder';
import { DermaLogo } from './components/DermaLogo';
import { HeroVisuals } from './components/HeroVisuals';
import { MatchedCasesPreview } from './components/MatchedCasesPreview';
import { PrivacyNotice } from './components/PrivacyNotice';
import { ResultCards } from './components/ResultCards';
import { SafetyNotice } from './components/SafetyNotice';
import { WizardView } from './components/WizardView';
import type { IntakeFormValues } from './types/intake';
import './styles.css';

function Results({ result }: { result: PredictResponse }) {
  return (
    <section className="results" aria-live="polite" aria-labelledby="results-heading">
      <h2 id="results-heading">Your treatment response estimate</h2>
      <p className="hint">
        Estimates are derived from visually similar reference before/after cases.
        This is decision-support, not a diagnosis.
      </p>
      {result.exact_match && <ExactMatchBanner match={result.exact_match} />}
      <ResultCards likelihoods={result.likelihoods} />
      <ExplanationPanel explanation={result.explanation} />
      {result.heatmap.overlay_url ? <HeatmapPlaceholder heatmap={result.heatmap} /> : null}
      <MatchedCasesPreview matchedPatients={result.matched_patients} />
      <div className="notice-footer">
        <SafetyNotice copy={result.disclaimer} compact />
        <PrivacyNotice copy={result.privacy_notice} compact />
      </div>
    </section>
  );
}

export default function App() {
  const [view, setView] = useState<'landing' | 'wizard'>('landing');
  const [result, setResult] = useState<PredictResponse | null>(null);

  const mutation = useMutation({
    mutationFn: ({ values, image }: { values: IntakeFormValues; image: File }) => predict(values, image),
    onSuccess: setResult,
  });

  function onSubmit(values: IntakeFormValues, selectedFile: File) {
    mutation.mutate({ values, image: selectedFile });
  }

  function restart() {
    setResult(null);
    mutation.reset();
  }

  if (view === 'wizard') {
    return (
      <main className="app-shell">
        <WizardView
          onSubmit={onSubmit}
          isSubmitting={mutation.isPending}
          errorMessage={mutation.isError ? mutation.error.message : undefined}
          hasResult={result !== null}
          resultsSlot={result ? <Results result={result} /> : null}
          onExit={() => {
            restart();
            setView('landing');
          }}
          onRestart={restart}
        />
      </main>
    );
  }

  return (
    <main className="app-shell">
      {/* Hero Section: Split Layout */}
      <section className="hero-section">
        <div className="hero-left">
          <div className="brand-row">
            <DermaLogo className="brand-logo" />
            <h1>DermaMatch</h1>
          </div>
          <p className="hero-tagline">Know Your Treatment Path</p>
          <p className="hero-description">
            Upload a baseline photo and discover how likely you are to improve on Dupixent or Ebglyss — before you meet your dermatologist.
          </p>
          <ul className="hero-features">
            <li>Explainable predictions based on similar patients</li>
            <li>See which biomarkers align with your skin</li>
            <li>Understand your response likelihood</li>
          </ul>
          <div className="hero-cta">
            <button type="button" className="cta-button" onClick={() => setView('wizard')}>
              Get started →
            </button>
          </div>
        </div>
        <div className="hero-right">
          <HeroVisuals />
        </div>
      </section>

      <div className="notice-footer">
        <SafetyNotice compact />
        <PrivacyNotice compact />
      </div>
    </main>
  );
}
