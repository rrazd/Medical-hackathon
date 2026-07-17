import type { ExactMatch } from '../api/dermaMatchClient';

type ExactMatchBannerProps = {
  match: ExactMatch;
};

export function ExactMatchBanner({ match }: ExactMatchBannerProps) {
  return (
    <section className="exact-match" role="status" aria-labelledby="exact-match-heading">
      <div className="exact-match-badge">✓ Exact image match</div>
      <h2 id="exact-match-heading">
        We found a reference photo that looks <strong>identical</strong> to yours.
      </h2>
      <p>
        Your upload matches case <strong>{match.case_id}</strong>, who improved on{' '}
        <strong>{match.biologic}</strong>. Because it is the same skin presentation, we are highly
        confident <strong>{match.biologic}</strong> is your closest-matched treatment.
      </p>
      {match.before_image_url && match.after_image_url && (
        <div className="exact-match-images">
          <figure>
            <img src={match.before_image_url} alt={`${match.case_id} baseline (before)`} />
            <figcaption>Matched before</figcaption>
          </figure>
          <figure>
            <img src={match.after_image_url} alt={`${match.case_id} after treatment`} />
            <figcaption>After {match.biologic}</figcaption>
          </figure>
        </div>
      )}
    </section>
  );
}
