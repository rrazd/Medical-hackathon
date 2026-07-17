import type { MatchedPatient } from '../api/dermaMatchClient';

type MatchedCasesPreviewProps = {
  matchedPatients: MatchedPatient[];
};

export function MatchedCasesPreview({ matchedPatients }: MatchedCasesPreviewProps) {
  return (
    <section className="card" aria-labelledby="matched-cases-heading">
      <h2 id="matched-cases-heading">Patients like you</h2>
      <p className="hint">
        De-identified reference cases whose baseline skin most closely matches your photo.
      </p>

      <div className="matched-case-list">
        {matchedPatients.map((match) => (
          <article className="matched-case" key={match.case_id}>
            <h3>{match.case_id}</h3>
            <p>
              {Math.round(match.similarity * 100)}% similarity · improved on{' '}
              {match.biologic_used}
            </p>
            <p>{match.demographic_summary}</p>
            <ul>
              {match.matching_reasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
            {match.before_image_url && match.after_image_url ? (
              <div className="matched-case-images">
                <figure>
                  <img src={match.before_image_url} alt={`${match.case_id} baseline (before)`} loading="lazy" />
                  <figcaption>Before</figcaption>
                </figure>
                <figure>
                  <img src={match.after_image_url} alt={`${match.case_id} after treatment`} loading="lazy" />
                  <figcaption>After {match.biologic_used}</figcaption>
                </figure>
              </div>
            ) : (
              <p className="hint">Before/after images are not available for this case.</p>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}
