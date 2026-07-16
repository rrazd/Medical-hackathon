const defaultPrivacyCopy =
  'Uploaded images are analyzed for this session only and are not stored as an account or EHR record.';

export function PrivacyNotice({ copy = defaultPrivacyCopy, compact = false }: { copy?: string; compact?: boolean }) {
  return (
    <section className={`notice privacy${compact ? ' notice--compact' : ''}`} aria-labelledby="privacy-heading">
      <h2 id="privacy-heading">Privacy notice</h2>
      <p>{copy}</p>
    </section>
  );
}
