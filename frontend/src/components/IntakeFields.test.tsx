import { zodResolver } from '@hookform/resolvers/zod';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useForm } from 'react-hook-form';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { IntakeFields } from './IntakeFields';
import { intakeSchema, type IntakeFormValues } from '../types/intake';

const emptyDefaults: IntakeFormValues = {
  age: undefined as unknown as number,
  sex: '',
  race_ethnicity: '',
  body_area: '',
  eczema_duration: '',
  itch_severity: '',
  atopic_comorbidities: '',
  tried_biologics: '',
  biologics_stopped_reason: '',
  nonbiologic_treatments: '',
  daily_routine: '',
};

function IntakeHarness({ onSubmit }: { onSubmit: (values: IntakeFormValues) => void }) {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<IntakeFormValues>({
    resolver: zodResolver(intakeSchema),
    defaultValues: emptyDefaults,
  });

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <IntakeFields register={register} errors={errors} />
      <button type="submit">Submit intake</button>
    </form>
  );
}

describe('IntakeFields', () => {
  afterEach(() => {
    cleanup();
  });

  it('shows required messages for all demographics and context fields', async () => {
    const user = userEvent.setup();
    render(<IntakeHarness onSubmit={vi.fn()} />);

    await user.click(screen.getByRole('button', { name: /submit intake/i }));

    expect(await screen.findByText(/age is required/i)).toBeInTheDocument();
    expect(screen.getByText(/sex is required/i)).toBeInTheDocument();
    expect(screen.getByText(/race\/ethnicity is required/i)).toBeInTheDocument();
    expect(screen.getByText(/body area is required/i)).toBeInTheDocument();
    expect(screen.getByText(/eczema duration is required/i)).toBeInTheDocument();
    expect(screen.getByText(/itch severity is required/i)).toBeInTheDocument();
    expect(screen.getAllByText(/this answer is required/i).length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText(/treatment history is required/i)).toBeInTheDocument();
    expect(screen.getByText(/your typical day is required/i)).toBeInTheDocument();
  });

  it('submits snake_case intake values that match the multipart contract', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<IntakeHarness onSubmit={onSubmit} />);

    await user.type(screen.getByLabelText(/^age$/i), '36');
    await user.selectOptions(screen.getByLabelText(/^sex$/i), 'female');
    await user.selectOptions(screen.getByLabelText(/race\/ethnicity/i), 'hispanic-latino');
    await user.type(screen.getByLabelText(/body area/i), 'forearms');
    await user.selectOptions(
      screen.getByLabelText(/how long have you had eczema/i),
      '1-3 years',
    );
    await user.selectOptions(
      screen.getByLabelText(/rate the severity of your itch/i),
      'moderate',
    );
    await user.selectOptions(screen.getByLabelText(/asthma or hay fever/i), 'none');
    await user.selectOptions(screen.getByLabelText(/tried biologics before/i), 'no');
    await user.type(screen.getByLabelText(/typical day/i), 'desk work and evening runs');
    await user.type(
      screen.getByLabelText(/non-biologic.*treatment history/i),
      'topical steroids',
    );
    await user.click(screen.getByRole('button', { name: /submit intake/i }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit).toHaveBeenCalledWith(
      {
        age: 36,
        sex: 'female',
        race_ethnicity: 'hispanic-latino',
        body_area: 'forearms',
        eczema_duration: '1-3 years',
        itch_severity: 'moderate',
        atopic_comorbidities: 'none',
        tried_biologics: 'no',
        biologics_stopped_reason: '',
        nonbiologic_treatments: 'topical steroids',
        daily_routine: 'desk work and evening runs',
      },
      expect.anything(),
    );
  });
});
