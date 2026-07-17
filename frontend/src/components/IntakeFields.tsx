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
          Body area
          <input
            aria-invalid={Boolean(errors.body_area)}
            aria-describedby={errors.body_area ? 'body-area-error' : undefined}
            {...register('body_area')}
          />
          <FieldError id="body-area-error" message={errors.body_area?.message} />
        </label>

        <label className="field-span-2">
          <span className="field-label-row">Tell us about your typical day <span className="field-optional">(optional)</span></span>
          <span className="field-help">
            A quick sketch of your routine — work, screen time, travel, exercise, kids —
            helps us weigh each biologic's dosing convenience and side-effect profile for your lifestyle.
          </span>
          <textarea
            rows={3}
            placeholder="e.g. Long days at a computer, lots of travel for work, and I run outdoors most mornings."
            aria-invalid={Boolean(errors.daily_routine)}
            aria-describedby={errors.daily_routine ? 'daily-routine-error' : undefined}
            {...register('daily_routine')}
          />
          <FieldError id="daily-routine-error" message={errors.daily_routine?.message} />
        </label>

        <label className="field-span-2">
          <span className="field-label-row">Prior treatments <span className="field-optional">(optional)</span></span>
          <span className="field-help">
            Any creams, pills, biologics, or therapies you've already tried for your skin.
          </span>
          <textarea
            rows={3}
            placeholder="e.g. Topical steroids, then methotrexate for 6 months."
            aria-invalid={Boolean(errors.prior_treatments)}
            aria-describedby={errors.prior_treatments ? 'prior-treatments-error' : undefined}
            {...register('prior_treatments')}
          />
          <FieldError id="prior-treatments-error" message={errors.prior_treatments?.message} />
        </label>
      </div>
    </fieldset>
  );
}
