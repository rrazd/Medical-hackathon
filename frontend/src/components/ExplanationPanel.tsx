import type { Explanation } from '../api/dermaMatchClient';

type ExplanationPanelProps = {
  explanation: Explanation;
};

export function ExplanationPanel({ explanation }: ExplanationPanelProps) {
  return (
    <section className="card" aria-labelledby="explanation-heading">
      <h2 id="explanation-heading">Why this estimate looks this way</h2>
      <p>{explanation.summary}</p>

      <div className="driver-list" role="list" aria-label="Top contributing biomarker drivers">
        {explanation.top_contributing_biomarkers.map((driver) => (
          <article className="driver-row" role="listitem" key={driver.name}>
            <div>
              <h3>{driver.label}</h3>
              <p className="hint">{driver.name}</p>
            </div>
            <dl className="driver-metrics">
              <div>
                <dt>Patient value</dt>
                <dd>{driver.patient_value}</dd>
              </div>
              <div>
                <dt>Direction</dt>
                <dd>{driver.direction}</dd>
              </div>
              <div>
                <dt>Driver weight</dt>
                <dd>{driver.weight.toFixed(2)}</dd>
              </div>
            </dl>
          </article>
        ))}
      </div>
    </section>
  );
}
