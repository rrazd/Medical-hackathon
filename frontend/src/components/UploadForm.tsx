import { zodResolver } from '@hookform/resolvers/zod';
import { useCallback, useRef, useState } from 'react';
import { useForm } from 'react-hook-form';

import { IntakeFields } from './IntakeFields';
import { intakeSchema, type IntakeFormValues } from '../types/intake';

type UploadFormProps = {
  onSubmit: (values: IntakeFormValues, image: File) => void;
  isSubmitting?: boolean;
  errorMessage?: string;
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

export function UploadForm({ onSubmit, isSubmitting = false, errorMessage }: UploadFormProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState('');
  const [fileError, setFileError] = useState('');
  const latestPreviewUrl = useRef<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<IntakeFormValues>({
    resolver: zodResolver(intakeSchema),
    defaultValues: emptyDefaults,
  });

  const formRef = useCallback((node: HTMLFormElement | null) => {
    if (node === null) {
      if (latestPreviewUrl.current) {
        URL.revokeObjectURL(latestPreviewUrl.current);
        latestPreviewUrl.current = null;
      }
    }
  }, []);

  function onFileChange(file: File | undefined) {
    if (!file) return;
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
      setSelectedFile(null);
      setFileName('');
      if (latestPreviewUrl.current) {
        URL.revokeObjectURL(latestPreviewUrl.current);
      }
      latestPreviewUrl.current = null;
      setPreviewUrl(null);
      setFileError('Upload must be a JPEG or PNG image.');
      return;
    }

    if (latestPreviewUrl.current) {
      URL.revokeObjectURL(latestPreviewUrl.current);
    }
    const nextPreviewUrl = URL.createObjectURL(file);
    latestPreviewUrl.current = nextPreviewUrl;
    setSelectedFile(file);
    setFileName(file.name);
    setPreviewUrl(nextPreviewUrl);
    setFileError('');
  }

  return (
    <form
      className="intake-form"
      ref={formRef}
      onSubmit={handleSubmit((values) => selectedFile && onSubmit(values, selectedFile))}
    >
      <fieldset>
        <legend>Photo upload</legend>
        <label htmlFor="image">Baseline AD photo (JPEG or PNG)</label>
        <input
          id="image"
          type="file"
          accept="image/png,image/jpeg"
          aria-describedby="image-help"
          onChange={(event) => onFileChange(event.target.files?.[0])}
        />
        <p className="hint" id="image-help">
          Choose a baseline photo to enable the mock estimate. The preview stays in this browser session only.
        </p>
        {fileError && (
          <p className="warning" role="alert">
            {fileError}
          </p>
        )}
        {fileName && <p>Selected file: {fileName}</p>}
        {previewUrl && (
          <div className="preview-panel">
            <img className="preview" src={previewUrl} alt="Selected baseline photo preview" />
          </div>
        )}
      </fieldset>

      <IntakeFields register={register} errors={errors} />

      {Object.keys(errors).length > 0 && (
        <p className="warning" role="alert">
          Please complete all required intake fields.
        </p>
      )}
      {!selectedFile && <p className="hint">Choose a photo to enable the mock estimate.</p>}
      <button type="submit" disabled={!selectedFile || isSubmitting}>
        {isSubmitting ? 'Estimating…' : 'Estimate response'}
      </button>
      {errorMessage && <p className="warning">{errorMessage}</p>}
    </form>
  );
}
