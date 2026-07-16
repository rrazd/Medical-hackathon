const defaultSafetyCopy =
  'DermaMatch is prototype decision-support for discussion with a dermatologist. It is not a diagnosis, prescription, or medical advice.';

export function SafetyNotice({ copy = defaultSafetyCopy, compact = false }: { copy?: string; compact?: boolean }) {
  return (
    <section className={`notice safety${compact ? ' notice--compact' : ''}`} aria-labelledby="safety-heading">
      <h2 id="safety-heading">Medical safety notice</h2>
      <p>{copy}</p>
    </section>
  );
}
