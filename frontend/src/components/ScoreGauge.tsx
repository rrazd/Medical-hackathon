import type { CSSProperties } from 'react';

type ScoreGaugeProps = {
  value: number; // 0–100
  label?: string;
  size?: number;
};

const ANGULAR_STEPS = 60;
const RAY_DOTS = 5;
const INNER_R = 80;
const OUTER_R = 118;

/**
 * Radial dot-burst gauge: a ring of rays made of shrinking dots, filled
 * clockwise from the top in proportion to `value`. Dots animate in with a
 * clockwise sweep on mount. Purely decorative rendering of a 0–100 score.
 */
export function ScoreGauge({ value, label = 'out of 100', size = 220 }: ScoreGaugeProps) {
  const fraction = Math.max(0, Math.min(100, value)) / 100;
  const rounded = Math.round(value);

  const dots = [];
  for (let step = 0; step < ANGULAR_STEPS; step += 1) {
    const angle = (step / ANGULAR_STEPS) * 2 * Math.PI - Math.PI / 2; // start at top
    const active = step / ANGULAR_STEPS <= fraction;
    for (let d = 0; d < RAY_DOTS; d += 1) {
      const t = d / (RAY_DOTS - 1);
      const r = INNER_R + t * (OUTER_R - INNER_R);
      const cx = 120 + r * Math.cos(angle);
      const cy = 120 + r * Math.sin(angle);
      const radius = 3.4 - t * 1.9;
      const opacity = active ? 1 - t * 0.3 : 0.28;
      const delay = step * 11 + d * 6; // clockwise sweep, outward per ray
      dots.push(
        <circle
          key={`${step}-${d}`}
          className="gauge-dot"
          cx={cx}
          cy={cy}
          r={radius}
          fill={active ? '#0f8768' : '#b6d9cd'}
          style={{ '--o': opacity, animationDelay: `${delay}ms`, transformOrigin: `${cx}px ${cy}px` } as CSSProperties}
        />,
      );
    }
  }

  return (
    <svg className="score-gauge" width={size} height={size} viewBox="0 0 240 240" role="img" aria-label={`${rounded} ${label}`}>
      <circle cx="120" cy="120" r="68" fill="#ffffff" stroke="#e3efe9" strokeWidth="1.5" />
      {dots}
      <text x="120" y="118" textAnchor="middle" className="score-gauge-value">
        {rounded}
      </text>
      <text x="120" y="142" textAnchor="middle" className="score-gauge-label">
        {label}
      </text>
    </svg>
  );
}
