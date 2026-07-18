import type { Explanation } from '../api/dermaMatchClient';

type ExplanationPanelProps = {
  explanation: Explanation;
};

export function ExplanationPanel({ explanation }: ExplanationPanelProps) {
  const considerations = explanation.lifestyle_considerations ?? [];

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

      {considerations.length > 0 && (
        <div className="considerations">
          <h3 className="considerations-subhead">What else we weighed for you</h3>
          <p className="considerations-intro">
            Beyond your photo, we factored in the personal details and side-effect concerns
            you shared. Here is exactly how each one influenced your recommendation — so you
            can see nothing you told us was ignored.
          </p>
          <ul className="considerations-list">
            {considerations.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
          <p className="considerations-footnote">
            These are decision-support considerations drawn from published data on both
            biologics — not a diagnosis or a guarantee. Review them with your dermatologist.
          </p>
        </div>
      )}
    </section>
  );
}
