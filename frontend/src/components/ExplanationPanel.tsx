import type { Explanation } from '../api/dermaMatchClient';

type ExplanationPanelProps = {
  explanation: Explanation;
};

export function ExplanationPanel({ explanation }: ExplanationPanelProps) {
  return (
    <section className="card" aria-labelledby="explanation-heading">
      <h2 id="explanation-heading">Why this estimate looks this way</h2>
      <p>{explanation.summary}</p>
      {explanation.recommendation_rationale && (
        <p className="recommendation-rationale">
          <strong>{explanation.recommendation_rationale}</strong>
        </p>
      )}
      <p>
        We compare visual biomarkers from your photo — lesion coverage, affected area,
        overall inflammation, redness intensity, skin texture, and dryness/scaling —
        against our curated database of real before/after cases to derive these results.
      </p>

      {explanation.lifestyle_considerations && explanation.lifestyle_considerations.length > 0 && (
        <div className="lifestyle-considerations">
          <h3>Lifestyle &amp; side-effect considerations</h3>
          <ul>
            {explanation.lifestyle_considerations.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
