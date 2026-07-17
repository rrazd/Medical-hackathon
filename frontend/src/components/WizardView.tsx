import { zodResolver } from '@hookform/resolvers/zod';
import { useCallback, useEffect, useRef, useState, type DragEvent, type ReactNode } from 'react';
import { useForm } from 'react-hook-form';

import { IntakeFields } from './IntakeFields';
import { intakeSchema, type IntakeFormValues } from '../types/intake';

type WizardViewProps = {
  onSubmit: (values: IntakeFormValues, image: File) => void;
  isSubmitting?: boolean;
  errorMessage?: string;
  hasResult: boolean;
  resultsSlot?: ReactNode;
  onExit: () => void;
  onRestart: () => void;
};

const emptyDefaults: IntakeFormValues = {
  age: undefined as unknown as number,
  sex: '',
  race_ethnicity: '',
  fitzpatrick_skin_type: '',
  body_area: '',
  prior_treatments: '',
  baseline_severity: '',
};

const INTAKE_STORAGE_KEY = 'dermamatch.intake';

function loadPersistedDefaults(): IntakeFormValues {
  try {
    const raw = localStorage.getItem(INTAKE_STORAGE_KEY);
    if (!raw) return emptyDefaults;
    const saved = JSON.parse(raw) as Partial<IntakeFormValues>;
    return { ...emptyDefaults, ...saved };
  } catch {
    return emptyDefaults;
  }
}

const STEPS = ['Upload photo', 'Your details', 'Review', 'Results'] as const;

function formatBytes(bytes: number): string {
  if (!bytes) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const fieldLabels: Record<keyof IntakeFormValues, string> = {
  age: 'Age',
  sex: 'Sex',
  race_ethnicity: 'Race/Ethnicity',
  fitzpatrick_skin_type: 'Fitzpatrick skin type',
  body_area: 'Body area',
  prior_treatments: 'Prior treatments',
  baseline_severity: 'Baseline severity',
};

export function WizardView({
  onSubmit,
  isSubmitting = false,
  errorMessage,
  hasResult,
  resultsSlot,
  onExit,
  onRestart,
}: WizardViewProps) {
  const [step, setStep] = useState(0);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState('');
  const [fileError, setFileError] = useState('');
  const [fileSize, setFileSize] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const analyzeStartRef = useRef(0);
  const latestPreviewUrl = useRef<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const {
    register,
    trigger,
    getValues,
    watch,
    formState: { errors },
  } = useForm<IntakeFormValues>({
    resolver: zodResolver(intakeSchema),
    defaultValues: loadPersistedDefaults(),
    mode: 'onSubmit',
    reValidateMode: 'onSubmit',
  });

  // Persist intake values so they survive a page refresh.
  useEffect(() => {
    const subscription = watch((values) => {
      try {
        localStorage.setItem(INTAKE_STORAGE_KEY, JSON.stringify(values));
      } catch {
        /* ignore storage failures */
      }
    });
    return () => subscription.unsubscribe();
  }, [watch]);

  // Advance to the results step once a prediction comes back — but hold the
  // "analyzing" animation for at least 3s so the analysis feels substantive.
  useEffect(() => {
    if (!hasResult || !isAnalyzing) return;
    const elapsed = Date.now() - analyzeStartRef.current;
    const remaining = Math.max(0, 3000 - elapsed);
    const timer = setTimeout(() => {
      setIsAnalyzing(false);
      setStep(3);
    }, remaining);
    return () => clearTimeout(timer);
  }, [hasResult, isAnalyzing]);

  // Stop the analyzing animation if the request fails so the error is visible.
  useEffect(() => {
    if (errorMessage) setIsAnalyzing(false);
  }, [errorMessage]);

  // Clean up object URL on unmount.
  useEffect(
    () => () => {
      if (latestPreviewUrl.current) URL.revokeObjectURL(latestPreviewUrl.current);
    },
    [],
  );

  const onFileChange = useCallback((file: File | undefined) => {
    if (!file) return;
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
      setSelectedFile(null);
      setFileName('');
      setFileSize(0);
      if (latestPreviewUrl.current) URL.revokeObjectURL(latestPreviewUrl.current);
      latestPreviewUrl.current = null;
      setPreviewUrl(null);
      setFileError('Upload must be a JPEG or PNG image.');
      return;
    }
    if (latestPreviewUrl.current) URL.revokeObjectURL(latestPreviewUrl.current);
    const nextPreviewUrl = URL.createObjectURL(file);
    latestPreviewUrl.current = nextPreviewUrl;
    setSelectedFile(file);
    setFileName(file.name);
    setFileSize(file.size);
    setPreviewUrl(nextPreviewUrl);
    setFileError('');
  }, []);

  const removeFile = useCallback(() => {
    if (latestPreviewUrl.current) URL.revokeObjectURL(latestPreviewUrl.current);
    latestPreviewUrl.current = null;
    setSelectedFile(null);
    setPreviewUrl(null);
    setFileName('');
    setFileSize(0);
    setFileError('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, []);

  const handleDragOver = useCallback((event: DragEvent) => {
    event.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((event: DragEvent) => {
    event.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault();
      setIsDragging(false);
      onFileChange(event.dataTransfer.files?.[0]);
    },
    [onFileChange],
  );

  async function goNext() {
    if (step === 0) {
      if (!selectedFile) {
        setFileError('Choose a baseline photo to continue.');
        return;
      }
      setStep(1);
      return;
    }
    if (step === 1) {
      const valid = await trigger();
      if (valid) setStep(2);
      return;
    }
  }

  function goStepBack() {
    setStep((s) => Math.max(0, s - 1));
  }

  function submit() {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    analyzeStartRef.current = Date.now();
    onSubmit(getValues(), selectedFile);
  }

  function restart() {
    if (latestPreviewUrl.current) URL.revokeObjectURL(latestPreviewUrl.current);
    latestPreviewUrl.current = null;
    setSelectedFile(null);
    setPreviewUrl(null);
    setFileName('');
    setFileSize(0);
    setFileError('');
    setStep(0);
    onRestart();
  }

  return (
    <section className="wizard" aria-labelledby="wizard-heading">
      <div className="wizard-header">
        <h2 id="wizard-heading" className="wizard-title">
          Get Your Personalized Estimate
        </h2>
      </div>

      {/* Stepper */}
      <ol className="stepper" aria-label="Progress">
        {STEPS.map((label, index) => {
          const state = index < step ? 'done' : index === step ? 'current' : 'upcoming';
          return (
            <li key={label} className={`step step--${state}`} aria-current={state === 'current' || undefined}>
              <span className="step-dot">{index < step ? '✓' : index + 1}</span>
              <span className="step-label">{label}</span>
            </li>
          );
        })}
      </ol>

      <div className="wizard-body card">
        {isAnalyzing && (
          <div className="analyzing" role="status" aria-live="polite">
            <div className="analyzing-scan" aria-hidden="true">
              {previewUrl && <img className="analyzing-photo" src={previewUrl} alt="" />}
              <div className="analyzing-scanline" />
              <div className="analyzing-ring" />
            </div>
            <h3>Analyzing your photo…</h3>
            <ul className="analyzing-steps">
              <li>Detecting affected skin regions</li>
              <li>Measuring visual biomarkers</li>
              <li>Matching against similar reference cases</li>
            </ul>
            <p className="hint">This only takes a moment.</p>
          </div>
        )}

        {/* Step 1 — Upload */}
        {!isAnalyzing && step === 0 && (
          <div className="wizard-step">
            <h3>Upload your baseline photo</h3>
            <p className="hint">A clear JPEG or PNG of the affected area. It stays in this browser session only.</p>

            <input
              ref={fileInputRef}
              id="wizard-image"
              className="visually-hidden-input"
              type="file"
              accept="image/png,image/jpeg"
              aria-label="Baseline AD photo"
              onChange={(event) => onFileChange(event.target.files?.[0])}
            />

            {!selectedFile ? (
              <label
                htmlFor="wizard-image"
                className={`file-drop${isDragging ? ' file-drop--drag' : ''}`}
                onDragOver={handleDragOver}
                onDragEnter={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <svg className="file-drop-icon" viewBox="0 0 48 48" width="48" height="48" aria-hidden="true">
                  <path
                    d="M24 32V14m0 0l-7 7m7-7l7 7"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path
                    d="M10 30v4a4 4 0 004 4h20a4 4 0 004-4v-4"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                <span className="file-drop-title">
                  {isDragging ? 'Drop your photo here' : 'Drag & drop or click to choose a photo'}
                </span>
                <span className="file-drop-sub">JPEG or PNG</span>
              </label>
            ) : (
              <div className="file-preview">
                <img className="file-preview-thumb" src={previewUrl ?? ''} alt="Selected baseline photo preview" />
                <div className="file-preview-meta">
                  <span className="file-preview-name">{fileName}</span>
                  <span className="file-preview-size">{formatBytes(fileSize)}</span>
                  <div className="file-preview-actions">
                    <button
                      type="button"
                      className="link-button"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      Change
                    </button>
                    <button type="button" className="link-button link-button--danger" onClick={removeFile}>
                      Remove
                    </button>
                  </div>
                </div>
              </div>
            )}

            {fileError && (
              <p className="warning" role="alert">
                {fileError}
              </p>
            )}
          </div>
        )}

        {/* Step 2 — Details */}
        {!isAnalyzing && step === 1 && (
          <div className="wizard-step">
            <h3>Tell us about you</h3>
            <p className="hint">These details help match you with similar reference cases.</p>
            <IntakeFields register={register} errors={errors} />
          </div>
        )}

        {/* Step 3 — Review */}
        {!isAnalyzing && step === 2 && (
          <div className="wizard-step">
            <h3>Review and confirm</h3>
            <p className="hint">Check your inputs, then run the estimate.</p>
            <div className="review-grid">
              {previewUrl && (
                <div className="review-photo">
                  <img className="preview" src={previewUrl} alt="Baseline photo preview" />
                  <p className="hint">{fileName}</p>
                </div>
              )}
              <dl className="review-list">
                {(Object.keys(fieldLabels) as (keyof IntakeFormValues)[]).map((key) => (
                  <div key={key}>
                    <dt>{fieldLabels[key]}</dt>
                    <dd>{String(getValues(key) ?? '—') || '—'}</dd>
                  </div>
                ))}
              </dl>
            </div>
            {errorMessage && <p className="warning">{errorMessage}</p>}
          </div>
        )}

        {/* Step 4 — Results */}
        {step === 3 && (
          <div className="wizard-step">
            {resultsSlot}
            <button type="button" className="secondary" onClick={restart}>
              Start over
            </button>
          </div>
        )}
      </div>

      {/* Footer navigation */}
      {!isAnalyzing && step < 3 && (
        <div className="wizard-nav">
          <div className="wizard-nav-left">
            {step > 0 && (
              <button type="button" className="secondary" onClick={goStepBack}>
                Back
              </button>
            )}
            <button type="button" className="tertiary" onClick={onExit}>
              Cancel
            </button>
          </div>
          {step < 2 && (
            <button type="button" onClick={goNext} disabled={step === 0 && !selectedFile}>
              Next
            </button>
          )}
          {step === 2 && (
            <button type="button" onClick={submit} disabled={isSubmitting}>
              {isSubmitting ? 'Estimating…' : 'Estimate response'}
            </button>
          )}
        </div>
      )}
    </section>
  );
}
