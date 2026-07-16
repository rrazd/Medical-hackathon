import type { Heatmap } from '../api/dermaMatchClient';

type HeatmapPlaceholderProps = {
  heatmap: Heatmap;
};

export function HeatmapPlaceholder({ heatmap }: HeatmapPlaceholderProps) {
  return (
    <section className="card" aria-labelledby="heatmap-heading">
      <h2 id="heatmap-heading">Biomarker heatmap placeholder</h2>
      <div className="placeholder-box" aria-label="No real heatmap overlay available in Phase 1">
        {heatmap.overlay_url ? (
          <img src={heatmap.overlay_url} alt="Mock biomarker heatmap overlay" />
        ) : (
          <p>No heatmap image is generated in Phase 1.</p>
        )}
      </div>
      <p>{heatmap.legend}</p>
    </section>
  );
}
