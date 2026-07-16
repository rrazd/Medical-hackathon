import type { CSSProperties } from 'react';

type DotBurstProps = {
  className?: string;
  size?: number;
};

const ANGULAR_STEPS = 72;
const RAY_DOTS = 5;
const INNER_R = 150;
const OUTER_R = 192;

/**
 * Purely decorative radial dot-burst ring, meant to frame content placed
 * at its center (e.g. the before/after cards). Dots animate in on mount.
 */
export function DotBurst({ className, size = 360 }: DotBurstProps) {
  const dots = [];
  for (let step = 0; step < ANGULAR_STEPS; step += 1) {
    const angle = (step / ANGULAR_STEPS) * 2 * Math.PI - Math.PI / 2;
    for (let d = 0; d < RAY_DOTS; d += 1) {
      const t = d / (RAY_DOTS - 1);
      const r = INNER_R + t * (OUTER_R - INNER_R);
      const cx = 200 + r * Math.cos(angle);
      const cy = 200 + r * Math.sin(angle);
      const radius = 3.2 - t * 1.8;
      // Gentle per-angle shimmer for an organic burst.
      const wobble = 0.15 * Math.sin(step * 0.7);
      const opacity = (0.85 - t * 0.4 + wobble);
      const delay = step * 9 + d * 5;
      dots.push(
        <circle
          key={`${step}-${d}`}
          className="gauge-dot"
          cx={cx}
          cy={cy}
          r={radius}
          fill="#14a37f"
          style={{ '--o': Math.max(0.12, opacity), animationDelay: `${delay}ms`, transformOrigin: `${cx}px ${cy}px` } as CSSProperties}
        />,
      );
    }
  }

  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 400 400"
      aria-hidden="true"
      focusable="false"
    >
      {dots}
    </svg>
  );
}
