import { useEffect, useRef, useState } from 'react';

import type { ExactMatch } from '../api/dermaMatchClient';

type ExactMatchToggletipProps = {
  label: string;
  match: ExactMatch;
};

export function ExactMatchToggletip({ label, match }: ExactMatchToggletipProps) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    if (!open) return;
    function onDocClick(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') setOpen(false);
    }
    document.addEventListener('mousedown', onDocClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return (
    <span className="toggletip" ref={containerRef}>
      <button
        type="button"
        className="toggletip-trigger"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        {label}
      </button>
      {open && (
        <span className="toggletip-bubble" role="status">
          <span className="exact-match-badge">✓ Exact image match</span>
          <span className="toggletip-text">
            Identical to case <strong>{match.case_id}</strong>, who improved on{' '}
            <strong>{match.biologic}</strong>.
          </span>
          {match.before_image_url && match.after_image_url && (
            <span className="toggletip-images">
              <span className="toggletip-figure">
                <img src={match.before_image_url} alt={`${match.case_id} before`} />
                <span>Before</span>
              </span>
              <span className="toggletip-figure">
                <img src={match.after_image_url} alt={`${match.case_id} after ${match.biologic}`} />
                <span>After</span>
              </span>
            </span>
          )}
        </span>
      )}
    </span>
  );
}
