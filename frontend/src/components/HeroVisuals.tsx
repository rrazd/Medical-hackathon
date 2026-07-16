import './HeroVisuals.css';
import { DotBurst } from './DotBurst';
import { SkinSwatch } from './SkinSwatch';

function SyringeIcon() {
  return (
    <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M18 2l4 4" />
      <path d="M15 5l4 4" />
      <path d="M17.5 6.5L8 16l-2.5 2.5" />
      <path d="M6.5 14.5l3 3" />
      <path d="M9 12l3 3" />
      <path d="M5.5 17.5L3 20l1 1 2.5-2.5" />
    </svg>
  );
}

function AntibodyIcon() {
  return (
    <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 21v-8" />
      <path d="M12 13L6 4" />
      <path d="M12 13l6-9" />
      <circle cx="5" cy="3" r="1.6" />
      <circle cx="19" cy="3" r="1.6" />
      <circle cx="12" cy="22" r="1.6" fill="currentColor" />
    </svg>
  );
}

export function HeroVisuals() {
  return (
    <div className="hero-visuals-container">
      {/* Soft organic blob for framing, behind everything */}
      <svg className="hero-blob" viewBox="0 0 600 600" aria-hidden="true">
        <defs>
          <linearGradient id="hero-blob-fill" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#dfe6ff" />
            <stop offset="55%" stopColor="#eaf1ff" />
            <stop offset="100%" stopColor="#e7f7f1" />
          </linearGradient>
        </defs>
        <path
          fill="url(#hero-blob-fill)"
          d="M441,101c54,38,104,84,120,142c16,58,-2,128,-40,181c-38,53,-96,89,-159,100c-63,11,-131,-3,-186,-38c-55,-35,-97,-92,-107,-153c-10,-61,12,-126,54,-176c42,-50,104,-85,167,-91c63,-6,97,-3,151,35Z"
        />
      </svg>

      {/* Background decorative arc */}
      <svg className="background-arc" viewBox="0 0 400 400" preserveAspectRatio="none" aria-hidden="true">
        <path
          d="M 50 100 Q 200 50 350 150"
          stroke="#3457d5"
          strokeWidth="2"
          fill="none"
          strokeDasharray="5,5"
          opacity="0.3"
        />
      </svg>

      {/* Centered stage keeps floating chips pinned to the visuals, never overlapping them */}
      <div className="visuals-stage">
        <div className="burst-wrap">
          {/* Dot-burst ring framing the before/after cards */}
          <DotBurst className="cards-burst" />

          <div className="visuals-main">
            <div className="example-card before-card">
              <div className="card-label">Before</div>
              <SkinSwatch variant="before" className="skin-simulation" />
              <p className="card-caption">Baseline AD</p>
            </div>

            <div className="example-card after-card">
              <div className="card-label">After</div>
              <SkinSwatch variant="after" className="skin-simulation" />
              <p className="card-caption">Response</p>
            </div>
          </div>
        </div>

        {/* Floating elements — positioned relative to the stage corners */}
        <div className="floating-element floating-1">
          <div className="biologic-icon injection" title="Injectable biologic">
            <SyringeIcon />
          </div>
        </div>

        <div className="floating-element floating-2">
          <div className="biologic-icon antibody" title="Monoclonal antibody">
            <AntibodyIcon />
          </div>
        </div>

        <div className="floating-element floating-3">
          <div className="treatment-icon dupixent">Dupixent</div>
        </div>

        <div className="floating-element floating-4">
          <div className="treatment-icon ebglyss">Ebglyss</div>
        </div>
      </div>
    </div>
  );
}
