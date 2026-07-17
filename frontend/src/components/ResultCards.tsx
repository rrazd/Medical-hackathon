import type { BiologicLikelihood } from '../api/dermaMatchClient';
import { ScoreGauge } from './ScoreGauge';

type ResultCardsProps = {
  likelihoods: BiologicLikelihood[];
};

export function ResultCards({ likelihoods }: ResultCardsProps) {
  return (
    <section className="card results-panel" aria-labelledby="likelihood-heading">
      <div className="section-heading">
        <p className="eyebrow">Prototype output</p>
        <h2 id="likelihood-heading">Dupixent vs Ebglyss likelihood comparison</h2>
        <p className="hint">Estimated from your photo against visually similar reference cases.</p>
      </div>

      <div className="result-grid" aria-label="Treatment response estimates">
        {likelihoods.map((item) => (
          <article className="result-card" key={item.biologic}>
            <h3>{item.biologic}</h3>
            <div className="result-gauge">
              <ScoreGauge value={item.likelihood_pct} label="% likelihood" />
            </div>
            <dl className="result-metrics">
              <div>
                <dt>Confidence label</dt>
                <dd>{item.confidence_label}</dd>
              </div>
              <div>
                <dt>Matched cases</dt>
                <dd>{item.matched_case_count} similar cases</dd>
              </div>
              <div>
                <dt>Match strength</dt>
                <dd>{item.weighted_outcome_score.toFixed(2)}</dd>
              </div>
            </dl>
            <p className="caveat">{item.caveat}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
