import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';

import { predict, type PredictResponse } from './api/dermaMatchClient';
import { ExplanationPanel } from './components/ExplanationPanel';
import { DermaLogo } from './components/DermaLogo';
import { HeroVisuals } from './components/HeroVisuals';
import { PrivacyNotice } from './components/PrivacyNotice';
import { ResultCards } from './components/ResultCards';
import { SafetyNotice } from './components/SafetyNotice';
import { WizardView, INTAKE_STORAGE_KEY } from './components/WizardView';
import type { IntakeFormValues } from './types/intake';
import './styles.css';

function Results({ result }: { result: PredictResponse }) {
  const recommended =
    result.exact_match?.biologic ??
    [...result.likelihoods].sort((a, b) => b.likelihood_pct - a.likelihood_pct)[0]?.biologic;
  return (
    <section className="results" aria-live="polite" aria-label="Your treatment response estimate">
      {recommended && (
        <div className="recommendation-hero" role="status">
          <span className="recommendation-label">Recommended for you to try</span>
          <strong className="recommendation-name">{recommended}</strong>
        </div>
      )}
      <ResultCards likelihoods={result.likelihoods} exactMatch={result.exact_match} />
      <ExplanationPanel explanation={result.explanation} />
    </section>
  );
}

export default function App() {
  const [view, setView] = useState<'landing' | 'wizard'>('landing');
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [wizardKey, setWizardKey] = useState(0);

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
    // Remount the wizard so useForm re-initializes from the freshly-cleared
    // storage — guarantees a blank step 2 even when Start over is pressed from
    // the results step (where the intake fields are unmounted).
    setWizardKey((key) => key + 1);
  }

  function beginWizard() {
    // Entering from the landing page is a fresh start — clear any persisted
    // intake so step 2 begins blank.
    try {
      localStorage.removeItem(INTAKE_STORAGE_KEY);
    } catch {
      /* ignore storage failures */
    }
    setResult(null);
    mutation.reset();
    setWizardKey((key) => key + 1);
    setView('wizard');
  }

  if (view === 'wizard') {
    return (
      <main className="app-shell">
        <WizardView
          key={wizardKey}
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
            <button type="button" className="cta-button" onClick={beginWizard}>
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
