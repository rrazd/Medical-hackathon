import type { FieldErrors, UseFormRegister } from 'react-hook-form';

import type { IntakeFormValues } from '../types/intake';

type IntakeFieldsProps = {
  register: UseFormRegister<IntakeFormValues>;
  errors: FieldErrors<IntakeFormValues>;
  triedBiologics?: string;
};

function FieldError({ id, message }: { id: string; message?: string }) {
  if (!message) return null;
  return (
    <span className="field-error" id={id} role="alert">
      {message}
    </span>
  );
}

export function IntakeFields({ register, errors, triedBiologics }: IntakeFieldsProps) {
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
          <select
            aria-invalid={Boolean(errors.race_ethnicity)}
            aria-describedby={errors.race_ethnicity ? 'race-ethnicity-error' : undefined}
            {...register('race_ethnicity')}
          >
            <option value="">Select race/ethnicity</option>
            <option value="asian">Asian</option>
            <option value="black">Black</option>
            <option value="hispanic-latino">Hispanic or Latino</option>
            <option value="white">White</option>
            <option value="other">Other</option>
            <option value="prefer-not-to-say">Prefer not to say</option>
          </select>
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

        <label>
          How long have you had eczema?
          <select
            aria-invalid={Boolean(errors.eczema_duration)}
            aria-describedby={errors.eczema_duration ? 'eczema-duration-error' : undefined}
            {...register('eczema_duration')}
          >
            <option value="">Select duration</option>
            <option value="<6 months">Less than 6 months</option>
            <option value="6-12 months">6–12 months</option>
            <option value="1-3 years">1–3 years</option>
            <option value="3-5 years">3–5 years</option>
            <option value="5+ years">More than 5 years</option>
          </select>
          <FieldError id="eczema-duration-error" message={errors.eczema_duration?.message} />
        </label>

        <label>
          Rate the severity of your itch
          <select
            aria-invalid={Boolean(errors.itch_severity)}
            aria-describedby={errors.itch_severity ? 'itch-severity-error' : undefined}
            {...register('itch_severity')}
          >
            <option value="">Select severity</option>
            <option value="none">None</option>
            <option value="mild">Mild</option>
            <option value="moderate">Moderate</option>
            <option value="severe">Severe</option>
            <option value="very-severe">Very severe</option>
          </select>
          <FieldError id="itch-severity-error" message={errors.itch_severity?.message} />
        </label>

        <label>
          Do you have asthma or hay fever?
          <select
            aria-invalid={Boolean(errors.atopic_comorbidities)}
            aria-describedby={errors.atopic_comorbidities ? 'atopic-comorbidities-error' : undefined}
            {...register('atopic_comorbidities')}
          >
            <option value="">Select an option</option>
            <option value="none">No</option>
            <option value="asthma">Yes — asthma</option>
            <option value="hay-fever">Yes — hay fever (allergic rhinitis)</option>
            <option value="both">Yes — both</option>
          </select>
          <FieldError id="atopic-comorbidities-error" message={errors.atopic_comorbidities?.message} />
        </label>

        <label>
          Have you tried biologics before?
          <select
            aria-invalid={Boolean(errors.tried_biologics)}
            aria-describedby={errors.tried_biologics ? 'tried-biologics-error' : undefined}
            {...register('tried_biologics')}
          >
            <option value="">Select an option</option>
            <option value="no">No</option>
            <option value="yes">Yes</option>
          </select>
          <FieldError id="tried-biologics-error" message={errors.tried_biologics?.message} />
        </label>

        {triedBiologics === 'yes' && (
          <label className="field-span-2">
            <span className="field-label-row">Why did you stop your previous biologic?</span>
            <textarea
              rows={3}
              placeholder="e.g. It stopped working after a year, or I had side effects like eye irritation."
              aria-invalid={Boolean(errors.biologics_stopped_reason)}
              aria-describedby={errors.biologics_stopped_reason ? 'biologics-stopped-reason-error' : undefined}
              {...register('biologics_stopped_reason')}
            />
            <FieldError
              id="biologics-stopped-reason-error"
              message={errors.biologics_stopped_reason?.message}
            />
          </label>
        )}

        <label className="field-span-2">
          <span className="field-label-row">Tell us about your typical day</span>
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
          <span className="field-label-row">Other (non-biologic) treatment history</span>
          <span className="field-help">
            Creams, pills, phototherapy, or other non-biologic therapies you've tried for your skin.
          </span>
          <textarea
            rows={3}
            placeholder="e.g. Topical steroids, then methotrexate for 6 months."
            aria-invalid={Boolean(errors.nonbiologic_treatments)}
            aria-describedby={errors.nonbiologic_treatments ? 'nonbiologic-treatments-error' : undefined}
            {...register('nonbiologic_treatments')}
          />
          <FieldError id="nonbiologic-treatments-error" message={errors.nonbiologic_treatments?.message} />
        </label>
      </div>
    </fieldset>
  );
}
