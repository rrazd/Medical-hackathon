import type { FieldErrors, UseFormRegister } from 'react-hook-form';

import type { IntakeFormValues } from '../types/intake';

type IntakeFieldsProps = {
  register: UseFormRegister<IntakeFormValues>;
  errors: FieldErrors<IntakeFormValues>;
};

function FieldError({ id, message }: { id: string; message?: string }) {
  if (!message) return null;
  return (
    <span className="field-error" id={id} role="alert">
      {message}
    </span>
  );
}

export function IntakeFields({ register, errors }: IntakeFieldsProps) {
  return (
    <fieldset>
      <legend>Demographics and context</legend>
      <div className="form-grid">
        <label>
          Age
          <input
            type="number"
            min="1"
            max="129"
            aria-invalid={Boolean(errors.age)}
            aria-describedby={errors.age ? 'age-error' : undefined}
            {...register('age')}
          />
          <FieldError id="age-error" message={errors.age?.message} />
        </label>

        <label>
          Sex
          <select
            aria-invalid={Boolean(errors.sex)}
            aria-describedby={errors.sex ? 'sex-error' : undefined}
            {...register('sex')}
          >
            <option value="">Select sex</option>
            <option value="female">Female</option>
            <option value="male">Male</option>
            <option value="nonbinary">Non-binary</option>
            <option value="prefer-not-to-say">Prefer not to say</option>
          </select>
          <FieldError id="sex-error" message={errors.sex?.message} />
        </label>

        <label>
          Race/Ethnicity
          <input
            aria-invalid={Boolean(errors.race_ethnicity)}
            aria-describedby={errors.race_ethnicity ? 'race-ethnicity-error' : undefined}
            {...register('race_ethnicity')}
          />
          <FieldError id="race-ethnicity-error" message={errors.race_ethnicity?.message} />
        </label>

        <label>
          Fitzpatrick skin type
          <select
            aria-invalid={Boolean(errors.fitzpatrick_skin_type)}
            aria-describedby={errors.fitzpatrick_skin_type ? 'fitzpatrick-skin-type-error' : undefined}
            {...register('fitzpatrick_skin_type')}
          >
            <option value="">Select type</option>
            {['I', 'II', 'III', 'IV', 'V', 'VI'].map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
          <FieldError id="fitzpatrick-skin-type-error" message={errors.fitzpatrick_skin_type?.message} />
        </label>

        <label>
          Body area
          <input
            aria-invalid={Boolean(errors.body_area)}
            aria-describedby={errors.body_area ? 'body-area-error' : undefined}
            {...register('body_area')}
          />
          <FieldError id="body-area-error" message={errors.body_area?.message} />
        </label>

        <label>
          Prior treatments
          <input
            aria-invalid={Boolean(errors.prior_treatments)}
            aria-describedby={errors.prior_treatments ? 'prior-treatments-error' : undefined}
            placeholder="Use none if there have been no prior treatments"
            {...register('prior_treatments')}
          />
          <FieldError id="prior-treatments-error" message={errors.prior_treatments?.message} />
        </label>

        <label>
          Baseline severity
          <select
            aria-invalid={Boolean(errors.baseline_severity)}
            aria-describedby={errors.baseline_severity ? 'baseline-severity-error' : undefined}
            {...register('baseline_severity')}
          >
            <option value="">Select severity</option>
            <option value="mild">Mild</option>
            <option value="moderate">Moderate</option>
            <option value="severe">Severe</option>
          </select>
          <FieldError id="baseline-severity-error" message={errors.baseline_severity?.message} />
        </label>
      </div>
    </fieldset>
  );
}
