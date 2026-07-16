import { useId } from 'react';

type SkinSwatchProps = {
  variant: 'before' | 'after';
  className?: string;
};

/**
 * Illustrative (not clinical) skin rendering.
 * `before` shows inflamed, dry, scaly atopic-dermatitis-like skin;
 * `after` shows calmer skin with faint residual discoloration.
 */
export function SkinSwatch({ variant, className }: SkinSwatchProps) {
  const uid = useId().replace(/[:]/g, '');
  const before = variant === 'before';

  const baseTop = '#f1c7a6';
  const baseBottom = before ? '#e2a888' : '#eec2a2';

  return (
    <svg
      className={className}
      viewBox="0 0 100 100"
      role="img"
      aria-label={before ? 'Illustration of inflamed, scaly skin' : 'Illustration of calmer, improved skin'}
      preserveAspectRatio="xMidYMid slice"
    >
      <defs>
        <radialGradient id={`base-${uid}`} cx="45%" cy="35%" r="80%">
          <stop offset="0%" stopColor={baseTop} />
          <stop offset="100%" stopColor={baseBottom} />
        </radialGradient>

        {/* Organic skin texture */}
        <filter id={`skin-${uid}`}>
          <feTurbulence
            type="fractalNoise"
            baseFrequency={before ? '0.9' : '0.65'}
            numOctaves="2"
            seed={before ? 7 : 3}
            result="noise"
          />
          <feColorMatrix
            in="noise"
            type="matrix"
            values={
              before
                ? '0 0 0 0 0.55  0 0 0 0 0.28  0 0 0 0 0.22  0 0 0 0.16 0'
                : '0 0 0 0 0.85  0 0 0 0 0.65  0 0 0 0 0.55  0 0 0 0.06 0'
            }
          />
        </filter>

        {/* Displacement to break up hard edges */}
        <filter id={`warp-${uid}`}>
          <feTurbulence type="fractalNoise" baseFrequency="0.08" numOctaves="2" seed={11} result="w" />
          <feDisplacementMap in="SourceGraphic" in2="w" scale={before ? 7 : 4} />
        </filter>

        <radialGradient id={`patch-${uid}`} cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor={before ? '#c0392b' : '#e6a58f'} stopOpacity={before ? 0.85 : 0.4} />
          <stop offset="55%" stopColor={before ? '#d9534f' : '#e9b49f'} stopOpacity={before ? 0.5 : 0.2} />
          <stop offset="100%" stopColor="#d9534f" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* Base skin */}
      <rect x="0" y="0" width="100" height="100" fill={`url(#base-${uid})`} />

      {/* Inflamed / residual patches */}
      <g filter={`url(#warp-${uid})`}>
        {before ? (
          <>
            <ellipse cx="34" cy="40" rx="26" ry="20" fill={`url(#patch-${uid})`} />
            <ellipse cx="66" cy="62" rx="24" ry="19" fill={`url(#patch-${uid})`} />
            <ellipse cx="72" cy="26" rx="15" ry="12" fill={`url(#patch-${uid})`} />
            <ellipse cx="24" cy="74" rx="16" ry="12" fill={`url(#patch-${uid})`} />
          </>
        ) : (
          <>
            <ellipse cx="38" cy="44" rx="20" ry="15" fill={`url(#patch-${uid})`} />
            <ellipse cx="68" cy="60" rx="16" ry="12" fill={`url(#patch-${uid})`} />
          </>
        )}
      </g>

      {/* Dry scaling flecks (mostly on 'before') */}
      {before && (
        <g fill="#f6ede4" opacity="0.75">
          <circle cx="30" cy="36" r="1.4" />
          <circle cx="40" cy="45" r="1.1" />
          <circle cx="26" cy="48" r="1.2" />
          <circle cx="63" cy="58" r="1.3" />
          <circle cx="70" cy="66" r="1.1" />
          <circle cx="58" cy="66" r="1" />
          <circle cx="74" cy="28" r="1.2" />
          <circle cx="22" cy="72" r="1.1" />
        </g>
      )}

      {/* Texture overlay */}
      <rect
        x="0"
        y="0"
        width="100"
        height="100"
        filter={`url(#skin-${uid})`}
        opacity={before ? 0.5 : 0.28}
        style={{ mixBlendMode: 'multiply' }}
      />

      {/* Soft top sheen */}
      <ellipse cx="40" cy="22" rx="42" ry="20" fill="#ffffff" opacity="0.12" />
    </svg>
  );
}
