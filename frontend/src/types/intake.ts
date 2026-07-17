import { z } from 'zod';

const requiredText = (label: string) => z.string().trim().min(1, `${label} is required.`);

export const intakeSchema = z.object({
  age: z.preprocess(
    (value) => (value === '' ? undefined : Number(value)),
    z
      .number({ required_error: 'Age is required.', invalid_type_error: 'Age is required.' })
      .int('Age must be a whole number.')
      .min(1, 'Age must be between 1 and 129.')
      .max(129, 'Age must be between 1 and 129.'),
  ),
  sex: requiredText('Sex'),
  race_ethnicity: requiredText('Race/ethnicity'),
  body_area: requiredText('Body area'),
  eczema_duration: requiredText('Eczema duration'),
  itch_severity: requiredText('Itch severity'),
  prior_treatments: requiredText('Prior treatments'),
  daily_routine: requiredText('Your typical day'),
});

export type IntakeFormValues = z.infer<typeof intakeSchema>;
