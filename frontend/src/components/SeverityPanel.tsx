import type { SeverityScores } from '../api/dermaMatchClient';

type SeverityPanelProps = {
  severity: SeverityScores;
};

export function SeverityPanel({ severity }: SeverityPanelProps) {
  const easiPct = Math.max(0, Math.min(100, (severity.easi / severity.easi_max) * 100));

  return (
    <section className="card severity-panel" aria-labelledby="severity-heading">
      <h2 id="severity-heading">Estimated baseline severity</h2>
      <div className="severity-metrics">
        <div className="severity-metric">
          <span className="severity-metric-label">IGA</span>
          <span className="severity-metric-value">
            {severity.iga}
            <span className="severity-metric-scale">/4</span>
          </span>
          <span className="severity-metric-note">{severity.iga_label}</span>
        </div>
        <div className="severity-metric">
          <span className="severity-metric-label">EASI</span>
          <span className="severity-metric-value">
            {Number.isInteger(severity.easi) ? severity.easi : severity.easi.toFixed(1)}
            <span className="severity-metric-scale">/{severity.easi_max}</span>
          </span>
          <span className="severity-metric-bar" aria-hidden="true">
            <span className="severity-metric-bar-fill" style={{ width: `${easiPct}%` }} />
          </span>
        </div>
      </div>
      <p className="severity-caption">
        IGA (Investigator&apos;s Global Assessment, 0–4) and EASI (Eczema Area and Severity
        Index, 0–72) are standard dermatology severity scores, estimated here from your
        photo&apos;s visual biomarkers. They approximate — and do not replace — a
        clinician&apos;s in-person scoring.
      </p>
    </section>
  );
}
