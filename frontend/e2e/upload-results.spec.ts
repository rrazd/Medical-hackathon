import { expect, test } from '@playwright/test';

const tinyPng = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=',
  'base64',
);

test('patient can upload/re-upload photo and see mock Dupixent and Ebglyss results with safety and privacy framing', async ({ page }) => {
  await page.goto('/');

  await expect(page.getByText(/not a diagnosis, prescription, or medical advice/i)).toBeVisible();
  await expect(page.getByText(/privacy/i)).toBeVisible();
  await expect(page.getByText(/not stored as an account or EHR record/i)).toBeVisible();

  const photoInput = page.getByLabel(/baseline AD photo/i);
  await photoInput.setInputFiles({ name: 'baseline-first.png', mimeType: 'image/png', buffer: tinyPng });
  await expect(page.getByRole('img', { name: /selected baseline photo preview/i })).toBeVisible();
  await expect(page.getByText(/baseline-first\.png/i)).toBeVisible();

  await photoInput.setInputFiles({ name: 'baseline-reupload.png', mimeType: 'image/png', buffer: tinyPng });
  await expect(page.getByText(/baseline-reupload\.png/i)).toBeVisible();

  await page.getByLabel(/age/i).fill('36');
  await page.getByLabel(/sex/i).selectOption('female');
  await page.getByLabel(/race\/ethnicity/i).fill('Latina');
  await page.getByLabel(/Fitzpatrick skin type/i).selectOption('IV');
  await page.getByLabel(/body area/i).fill('forearms');
  await page.getByLabel(/prior treatments/i).fill('topical steroids, moisturizer');
  await page.getByLabel(/baseline severity/i).selectOption('moderate');

  await page.getByRole('button', { name: /estimate response/i }).click();

  await expect(page.getByRole('heading', { name: /mock treatment response estimates/i })).toBeVisible();
  await expect(page.getByRole('heading', { name: /dupixent vs ebglyss likelihood comparison/i })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Dupixent', exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Ebglyss', exact: true })).toBeVisible();
  await expect(page.getByText(/72% mock likelihood/i)).toBeVisible();
  await expect(page.getByText(/64% mock likelihood/i)).toBeVisible();
  await expect(page.getByText(/weighted outcome score/i).first()).toBeVisible();
  await expect(page.getByText(/Mock explanation/i)).toBeVisible();
  await expect(page.getByRole('heading', { name: /biomarker heatmap placeholder/i })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'MOCK-001' })).toBeVisible();
  await expect(page.getByText(/not a diagnosis, prescription, or medical advice/i)).toHaveCount(2);
  await expect(page.getByText(/not stored as an account or EHR record/i)).toHaveCount(2);
});
