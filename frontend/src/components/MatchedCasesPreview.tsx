import type { MatchedPatient } from '../api/dermaMatchClient';

type MatchedCasesPreviewProps = {
  matchedPatients: MatchedPatient[];
};

export function MatchedCasesPreview({ matchedPatients }: MatchedCasesPreviewProps) {
  return (
    <section className="card" aria-labelledby="matched-cases-heading">
      <h2 id="matched-cases-heading">Matched-case preview stubs</h2>
      <p className="hint">
        These are mock “patients like you” placeholders. Real de-identified before/after cases come later.
      </p>

      <div className="matched-case-list">
        {matchedPatients.map((match) => (
          <article className="matched-case" key={match.case_id}>
            <h3>{match.case_id}</h3>
            <p>
              {Math.round(match.similarity * 100)}% mock similarity · {match.biologic_used} ·{' '}
              {match.outcome_label} ({match.outcome_score.toFixed(2)} outcome score)
            </p>
            <p>{match.demographic_summary}</p>
            <ul>
              {match.matching_reasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
            <p className="hint">Before/after images are not available in this Phase 1 stub.</p>
          </article>
        ))}
      </div>
    </section>
  );
}
