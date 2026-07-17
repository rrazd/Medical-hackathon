import type { BiologicLikelihood, ExactMatch } from '../api/dermaMatchClient';
import { ExactMatchToggletip } from './ExactMatchToggletip';
import { ScoreGauge } from './ScoreGauge';

type ResultCardsProps = {
  likelihoods: BiologicLikelihood[];
  exactMatch?: ExactMatch | null;
};

export function ResultCards({ likelihoods, exactMatch }: ResultCardsProps) {
  return (
    <section className="results-panel" aria-labelledby="likelihood-heading">
      <div className="section-heading">
        <h2 id="likelihood-heading">Dupixent vs Ebglyss comparison</h2>
        <p className="hint">
          Estimated from your photo against visually similar reference cases and taking into
          account lifestyle and side effect considerations.
        </p>
      </div>

      <div className="result-grid" aria-label="Treatment response estimates">
        {likelihoods.map((item) => {
          const isExact = exactMatch != null && item.biologic === exactMatch.biologic;
          return (
            <article className="result-card" key={item.biologic}>
              <h3>{item.biologic}</h3>
              <div className="result-gauge">
                <ScoreGauge value={item.likelihood_pct} label="% likelihood" />
              </div>
              <dl className="result-metrics">
                <div>
                  <dt>Confidence</dt>
                  <dd>
                    {isExact ? (
                      <ExactMatchToggletip label={item.confidence_label} match={exactMatch} />
                    ) : (
                      item.confidence_label
                    )}
                  </dd>
                </div>
              </dl>
            </article>
          );
        })}
      </div>
    </section>
  );
}
