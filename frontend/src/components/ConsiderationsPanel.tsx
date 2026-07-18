import type { Explanation } from '../api/dermaMatchClient';

type ConsiderationsPanelProps = {
  considerations?: string[];
};

export function ConsiderationsPanel({ considerations }: ConsiderationsPanelProps) {
  if (!considerations || considerations.length === 0) {
    return null;
  }

  return (
    <section className="card considerations-panel" aria-labelledby="considerations-heading">
      <div className="considerations-header">
        <span className="considerations-badge" aria-hidden="true">
          ✓
        </span>
        <div>
          <h2 id="considerations-heading">What we weighed for you</h2>
          <p className="considerations-intro">
            Beyond your photo, we factored in the personal details and side-effect concerns
            you shared. Here is exactly how each one influenced your recommendation — so you
            can see nothing you told us was ignored.
          </p>
        </div>
      </div>
      <ul className="considerations-list">
        {considerations.map((note) => (
          <li key={note}>{note}</li>
        ))}
      </ul>
      <p className="considerations-footnote">
        These are decision-support considerations drawn from published data on both
        biologics — not a diagnosis or a guarantee. Review them with your dermatologist.
      </p>
    </section>
  );
}
