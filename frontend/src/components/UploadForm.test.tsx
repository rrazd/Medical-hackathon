import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { UploadForm } from './UploadForm';

function stubObjectUrls() {
  Object.defineProperty(URL, 'createObjectURL', {
    configurable: true,
    writable: true,
    value: vi.fn((file: File) => `blob:${file.name}`),
  });
  Object.defineProperty(URL, 'revokeObjectURL', {
    configurable: true,
    writable: true,
    value: vi.fn(),
  });
}

describe('UploadForm', () => {
  beforeEach(() => {
    stubObjectUrls();
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it('shows a selected image preview and disables submit until an image exists', async () => {
    const user = userEvent.setup();
    render(<UploadForm onSubmit={vi.fn()} />);

    expect(screen.getByRole('button', { name: /estimate response/i })).toBeDisabled();
    expect(screen.getByText(/choose a photo to enable/i)).toBeInTheDocument();

    const image = new File(['first'], 'baseline-first.png', { type: 'image/png' });
    await user.upload(screen.getByLabelText(/baseline AD photo/i), image);

    expect(URL.createObjectURL).toHaveBeenCalledWith(image);
    expect(screen.getByRole('img', { name: /selected baseline photo preview/i })).toHaveAttribute(
      'src',
      'blob:baseline-first.png',
    );
    expect(screen.getByText(/selected file: baseline-first\.png/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /estimate response/i })).toBeEnabled();
  });

  it('replaces previews on re-upload and revokes old object URLs plus unmount cleanup', async () => {
    const user = userEvent.setup();
    render(<UploadForm onSubmit={vi.fn()} />);

    const input = screen.getByLabelText(/baseline AD photo/i);
    await user.upload(input, new File(['first'], 'baseline-first.png', { type: 'image/png' }));
    await user.upload(input, new File(['second'], 'baseline-reupload.png', { type: 'image/png' }));

    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:baseline-first.png');
    expect(screen.getByRole('img', { name: /selected baseline photo preview/i })).toHaveAttribute(
      'src',
      'blob:baseline-reupload.png',
    );
    expect(screen.getByText(/selected file: baseline-reupload\.png/i)).toBeInTheDocument();

    (URL.revokeObjectURL as unknown as { mockClear: () => void }).mockClear();
    cleanup();
    await waitFor(() => expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:baseline-reupload.png'));
    expect(URL.revokeObjectURL).toHaveBeenCalledTimes(1);
  });
});
