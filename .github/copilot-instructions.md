<!-- GSD:project-start source:PROJECT.md -->

## Project

**DermaMatch — Explainable AD Treatment Response Predictor**

DermaMatch is a browser-based prototype that helps atopic dermatitis (AD) patients understand
how likely they are to improve on a specific biologic *before* they meet their dermatologist.
A patient uploads a baseline photo and a few details about themselves; the tool analyzes visual
biomarkers, matches them against a curated reference dataset of real before/after cases, and
returns a transparent, explainable prediction of likely response for Dupixent (dupilumab) and
Ebglyss (lebrikizumab).

**Core Value:** Give a patient a trustworthy, *explainable* estimate of how much a given biologic is likely to
help them — grounded in similar real patients and visible skin biomarkers, never a black box.

### Constraints

- **Tech stack**: Python backend (FastAPI) for image biomarker analysis (OpenCV / scikit-image) and the matching engine; lightweight React frontend for upload → results. Chosen as the best fit for image + ML work in a real browser app.
- **Data**: Small, hand-curated local dataset. Methods must work well with limited data (favors similarity matching, not model training).
- **Medical safety**: Must present as decision-support with an explicit disclaimer; must not claim diagnostic certainty.
- **Privacy**: Only de-identified / consented images. No PII persistence beyond the working session.
- **Timeline**: Prototype-grade — demonstrable end-to-end flow prioritized over completeness.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->

## Technology Stack

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.12.x | Backend runtime for API, image processing, feature extraction, and matching | Current stable Python line with broad scientific-library support; faster than older 3.x versions and mature enough for hackathon delivery. Use 3.11.x only if a deployment target lacks 3.12 support. |
| FastAPI | ~0.115–0.116 | HTTP API for upload, analysis, matching, and result JSON | Standard modern Python API framework; excellent request validation, OpenAPI docs, async file upload support, and tight Pydantic integration. |
| Uvicorn | ~0.34–0.35 | ASGI server for local/dev serving | Lightweight default FastAPI server; simple `uvicorn app.main:app --reload` workflow for prototypes. |
| Pydantic | ~2.10–2.11 | Typed request/response models and config validation | Pydantic v2 is the current FastAPI baseline; useful for strict result schemas, patient metadata, and dataset row validation. |
| React | 19.x | Browser UI for upload, patient inputs, results, overlays, and comparison cards | Current React generation; suitable for a lightweight single-page prototype with rich image-result interactions. |
| Vite | 6.x or 7.x | Frontend build/dev server | Fastest practical React starter for hackathons; minimal configuration and excellent TypeScript support. Use Node 20.19+ if choosing Vite 7. |
| TypeScript | ~5.7–5.9 | Frontend type safety | Prevents schema drift between API responses and UI rendering; especially useful for biomarker/explanation payloads. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Pillow | ~11.x | Image loading, EXIF orientation handling, safe conversion to RGB, thumbnail generation | First step for uploaded images before handing arrays to OpenCV/scikit-image. |
| opencv-python-headless | ~4.10–4.12 | Classical CV operations: color conversion, thresholding, morphology, contour extraction, masks | Use backend/headless package, not GUI OpenCV. Primary tool for preprocessing, lesion-region masks, redness maps, and overlay-mask generation. |
| scikit-image | ~0.24–0.25 | Texture and image-feature algorithms | Use for GLCM/gray-level texture metrics, local binary patterns, region properties, filters, and color utilities where clearer than OpenCV. |
| numpy | ~2.1–2.2 | Numeric arrays and vector operations | Shared foundation for image arrays, biomarker vectors, normalization, and mask math. Pin with scientific stack to avoid binary incompatibilities. |
| scipy | ~1.14–1.15 | Scientific helpers, distance metrics, smoothing | Use for optional filtering, morphology helpers, and distance/statistical utilities. |
| scikit-learn | ~1.6–1.7 | Similarity matching via preprocessing and nearest neighbors | Use `StandardScaler`/`MinMaxScaler`, `OneHotEncoder`, `ColumnTransformer`, and `NearestNeighbors(metric='cosine')` or weighted Euclidean on normalized vectors. |
| pandas | ~2.2–2.3 | Local reference dataset loading and cleaning | Load curated CSV/XLSX, validate rows, join image paths/outcomes, and compute aggregate response likelihoods from matched cases. |
| openpyxl | ~3.1.x | XLSX reader/writer for pandas | Use only if the reference dataset is maintained as Excel. Prefer CSV for repo-friendly diffs. |
| python-multipart | ~0.0.12–0.0.20 | FastAPI multipart file uploads | Required for image upload endpoints. |
| pydantic-settings | ~2.7–2.9 | Environment/config model | Use for dataset path, upload limits, and demo toggles. |
| pytest | ~8.3–8.4 | Backend tests | Unit-test feature extraction on fixture images and matching on tiny synthetic datasets. |
| httpx | ~0.27–0.28 | FastAPI test client support / API testing | Use with pytest for endpoint tests. |
| shadcn/ui + Radix UI | Current CLI/components; Radix ~1.x | Accessible React components | Good hackathon fit: polished forms, cards, tabs, dialogs, and progress components without adopting a heavy design system. |
| Tailwind CSS | ~3.4 or 4.x | Utility CSS | Use with shadcn/ui. Choose Tailwind 3.4 for maximum plugin stability; Tailwind 4 if the team is comfortable with the newer setup. |
| React Hook Form | ~7.x | Form state for demographics/context | Use for patient metadata entry and validation with minimal rerenders. |
| Zod | ~3.24 or 4.x | Frontend schema validation | Mirror API result/input types at the UI boundary. Use the version expected by chosen form/resolver packages. |
| TanStack Query | ~5.x | API request state | Use for upload/analyze mutation, loading/error states, and cached result display. |
| Native Canvas API | Browser built-in | Biomarker heatmap overlay rendering | Best first choice: draw uploaded image, alpha-blended heatmap mask, and optional lesion contours on `<canvas>`. |
| react-konva | ~18/19-compatible current release | Higher-level canvas scene graph | Optional only if the overlay needs draggable annotations, layers, or interactive regions. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| uv | Python environment and dependency management | Recommended for fast, reproducible setup: `uv init`, `uv add`, `uv run`. Poetry is acceptable if already preferred. |
| Ruff | Python linting/formatting | Use as formatter + linter; replaces Black/isort/flake8 for this prototype. |
| mypy or pyright | Optional backend static typing | Use if time permits; prioritize tests over strict typing for hackathon pace. |
| pytest | Backend test runner | Add small fixture tests for color/texture features and nearest-neighbor ranking. |
| npm or pnpm | Frontend package manager | Use one consistently. pnpm is faster; npm is simpler and universally installed. |
| ESLint | Frontend linting | Use Vite React template defaults plus TypeScript rules. |
| Vitest | Frontend unit tests | Test result-formatting utilities and confidence/explanation rendering. |
| Playwright | Optional end-to-end smoke test | One happy-path upload/demo test if time allows; otherwise manual browser validation is acceptable. |
| Prettier | Frontend formatting | Use for TS/TSX/CSS consistency. |
| Git LFS | Optional image storage | Use only if real before/after demo images are large. Otherwise keep small de-identified demo images under `data/images/`. |

## Installation

# Backend, using uv from the repository root

# Frontend

# Optional UI scaffolding

## Biomarker Feature Extraction Pattern

## Similarity Matching Pattern

- Reference dataset row = demographics + baseline biomarker vector + biologic used + outcome + before/after image paths.
- Numeric features: age, erythema score, lesion coverage, texture metrics, dryness/scaling proxies, affected body area indicators.
- Categorical features: sex, race/ethnicity, background, prior treatments, body site, biologic.
- Preprocess with `ColumnTransformer`: scale numeric columns and one-hot encode categorical columns.
- Match with `sklearn.neighbors.NearestNeighbors`, starting with `metric='cosine'` on normalized feature vectors.
- Compute per-biologic likelihood from top-k weighted neighbors for that biologic: distance-weighted response rate, with small-sample confidence shown plainly.
- Explanation should list top contributing matched cases and top aligned biomarkers, not model-derived causality.

## Dataset Storage

| Data | Recommended Storage | Rationale |
|------|---------------------|-----------|
| Reference case metadata | `data/reference_cases.csv` as canonical source | Diffable, simple, reliable with pandas. |
| Spreadsheet editing | `.xlsx` maintained by humans, imported/exported via pandas + openpyxl | Useful for nontechnical curation; convert to CSV for app runtime. |
| Images | `data/images/<case_id>/before.jpg` and `after.jpg` | Simple local filesystem paths referenced from the dataset. |
| Derived features | `data/derived/features.parquet` or `features.csv` | Cache expensive image feature extraction; CSV is simplest, Parquet is better if available. |
| Uploads | In-memory or session-scoped local working directory | Prototype should avoid persistent PII. Do not retain patient uploads beyond the session/demo. |

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| FastAPI | Flask | Use Flask only if the team already has Flask scaffolding; FastAPI is better for typed schemas and auto-docs. |
| Uvicorn | Hypercorn/Gunicorn | Use Gunicorn+Uvicorn workers only for production-like deployment; unnecessary for local prototype. |
| OpenCV + scikit-image | Pure OpenCV | Fine if minimizing dependencies, but scikit-image makes texture/region feature code clearer. |
| Classical CV features | Deep learning segmentation/classification | Use only in a later validated product with labeled data, consent, compute, and clinical review. It contradicts the current prototype constraint. |
| scikit-learn NearestNeighbors | XGBoost/RandomForest/logistic regression | Use only after the dataset is large enough for train/validation splits and calibration. Not appropriate for a small curated case spreadsheet. |
| CSV + images on disk | SQLite | Use SQLite if multiple tables, audit logs, or local querying become painful; CSV is better for first curated dataset iteration. |
| Native Canvas | WebGL/Three.js | Use WebGL only for heavy interactive visualization; canvas is enough for 2D heatmap overlays. |
| shadcn/ui | Material UI | MUI is good for enterprise dashboards; shadcn is lighter and easier to customize for a polished hackathon demo. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Training a classifier on the small local dataset | High overfitting risk, poor calibration, weak explainability, and explicitly out of scope | Similar-patient matching with transparent feature vectors and neighbor examples. |
| Deep learning model training for image biomarkers | Requires labeled dermatology images, GPU time, privacy review, and validation; not feasible or allowed for this prototype | Classical CV with OpenCV/scikit-image and clearly labeled proxy biomarkers. |
| Cloud vision/medical AI APIs | Sends sensitive skin images to third parties and undermines local/de-identified prototype constraints | Local Python image processing only. |
| Storing identifiable uploads or accounts | Adds privacy/security scope that is explicitly out of scope | Session-only processing, de-identified reference images, no auth. |
| HIPAA/production claims | Prototype does not have compliance controls, auditability, clinical validation, or medical-device review | Prominent decision-support disclaimer. |
| Full PACS/EHR/FHIR integration | Unnecessary integration complexity for the demo | Local spreadsheet plus images on disk. |
| Heavy backend frameworks such as Django | Adds ORM/auth/admin complexity not needed for upload-analysis-results flow | FastAPI. |
| Next.js full-stack app as the primary backend | Python image/scientific stack is the center of gravity; duplicating backend logic in Node adds complexity | Vite React frontend + FastAPI backend. |
| MongoDB/Postgres for first prototype | Operational overhead without enough data volume or multi-user needs | CSV/XLSX + filesystem images. |
| Browser-only image analysis | Harder to reproduce, test, and share with matching code; limited scientific Python ecosystem | Backend Python feature extraction. |

## Stack Patterns by Variant

- Use CSV metadata, JPEG/PNG images on disk, FastAPI synchronous endpoints, native canvas overlays, and precomputed feature cache.
- Because this minimizes infrastructure and keeps the focus on the explainable flow.
- Keep scikit-learn preprocessing, cache feature matrices, and consider SQLite for metadata queries.
- Because transparent nearest-neighbor matching still works, but curation and filtering become easier with structured storage.
- Add explicit ROI/crop selection, color normalization cards where available, and body-site-specific feature normalization.
- Because classical erythema and scaling measures are highly sensitive to lighting, camera, and skin tone.
- Add `react-konva` for multiple overlay layers, annotation toggles, and interactive legends.
- Because native canvas is enough for alpha blending but less ergonomic for interactive layer management.
- Revisit consent, HIPAA/security architecture, dermatology label collection, fairness across skin tones, prospective validation, calibration, and medical-device regulatory constraints.
- Because the prototype stack is intentionally not production clinical infrastructure.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| FastAPI ~0.115–0.116 | Pydantic v2.x, Starlette current compatible range | Do not use Pydantic v1 patterns; use `BaseModel`, `model_config`, and `model_dump()`. |
| Uvicorn ~0.34–0.35 | Python 3.12, FastAPI ASGI app | Install `uvicorn[standard]` for local reload/websocket extras. |
| numpy ~2.1–2.2 | scikit-image ~0.24–0.25, scikit-learn ~1.6–1.7, pandas ~2.2–2.3 | Pin together in lockfile; avoid mixing old wheels compiled against NumPy 1.x. |
| opencv-python-headless ~4.10–4.12 | numpy 2.x | Prefer headless on servers/CI to avoid GUI/system-library issues. |
| scikit-learn ~1.6–1.7 | scipy ~1.14–1.15, numpy 2.x | Use `ColumnTransformer` + `NearestNeighbors`; persist preprocessing cautiously with matching version pins. |
| pandas ~2.2–2.3 | openpyxl ~3.1.x | `openpyxl` needed only for `.xlsx`; CSV does not require it. |
| Vite 6.x | Node 18/20+ | Conservative choice if local Node version is unknown. |
| Vite 7.x | Node 20.19+ or 22.12+ | Use only if the environment has a current Node runtime. |
| React 19.x | shadcn/Radix current releases | Verify any optional canvas library advertises React 19 compatibility. |
| Tailwind 4.x | Newer Vite plugin path | Tailwind 3.4 remains safer if following older shadcn tutorials. |

## Confidence Levels

| Area | Confidence | Reason |
|------|------------|--------|
| FastAPI/Pydantic/Uvicorn backend | HIGH | Mature, standard choice for Python API prototypes. |
| OpenCV/scikit-image/Pillow image pipeline | HIGH | Best fit for the explicit no-training classical CV constraint. |
| Specific biomarker proxy accuracy | MEDIUM | Technically feasible but lighting, skin tone, camera quality, and AD morphology limit reliability. |
| scikit-learn nearest-neighbor matching | HIGH | Directly matches the explainable small-data requirement. |
| Percentage likelihood interpretation | MEDIUM | Valid as a demo estimate from matched cases, but should show small-sample caveats. |
| CSV/XLSX + filesystem dataset | HIGH | Appropriate for a curated local prototype. |
| React/Vite/shadcn/canvas frontend | HIGH | Lightweight and suitable for upload/results/overlay UI. |
| Production clinical readiness | LOW | Out of scope; requires validation, compliance, security, and regulatory work. |

## Sources

- Project context: `/Users/rrazdan/workspace/hackathon/.planning/PROJECT.md` — requirements, constraints, scope, and key decisions.
- Stack template: `/Users/rrazdan/.copilot/gsd-core/templates/research-project/STACK.md` — required document structure.
- Current ecosystem knowledge as of 2025/2026 — package families, typical version ranges, compatibility caveats, and prototype architecture guidance. No web access used.

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.github/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
