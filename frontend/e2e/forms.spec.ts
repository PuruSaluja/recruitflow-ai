import { type Locator, type Page, expect, test } from "@playwright/test";

// How a form row lays its fields out. Nothing here reads or writes a record,
// so these tests need the built client and no data.

async function box(locator: Locator) {
  const found = await locator.boundingBox();
  expect(found).not.toBeNull();
  return found!;
}

/** The label with exactly this text, and the control it names. */
async function field(page: Page, name: string) {
  const id = await page.locator("label", { hasText: new RegExp(`^${name}$`) }).getAttribute("for");
  // By id, so both are still found once a test rewrites the label.
  return { label: page.locator(`label[for="${id}"]`), control: page.locator(`[id="${id}"]`) };
}

test.beforeEach(async ({ page }) => {
  // Wide enough for three fields on a line, on the phone project as well.
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto("/jobs/new");
  await expect(page.getByRole("heading", { name: "New job" })).toBeVisible();
});

test("keeps an input beside a field with a hint at its own height", async ({ page }) => {
  const status = await field(page, "Status");
  // Alone on its line at this width, so nothing beside it can stretch it.
  const alone = await box((await field(page, "Minimum years of experience")).control);

  for (const name of ["Salary from", "Salary to"]) {
    const salary = await box((await field(page, name)).control);
    // On Status's line, whose hint makes that field the taller one.
    expect(salary.y, name).toBeCloseTo((await box(status.control)).y, 0);
    expect(salary.height, name).toBeCloseTo(alone.height, 1);
  }
});

test("starts the controls on a line level when one label wraps", async ({ page }) => {
  const department = await field(page, "Department");
  const location = await field(page, "Location");
  // Long enough to wrap in any font.
  await department.label.evaluate((label) => {
    label.textContent = "Department ".repeat(12);
  });
  const wrapped = await box(department.label);
  const short = await box(location.label);
  expect(wrapped.height).toBeGreaterThan(short.height * 1.5);

  const beside = await box(location.control);
  expect(beside.y).toBeCloseTo((await box(department.control)).y, 1);
  // The one-line label stays with its control, not at the top of the line.
  expect(short.y + short.height).toBeCloseTo(wrapped.y + wrapped.height, 1);
  expect(beside.y).toBeGreaterThan(short.y + short.height);
  expect(beside.y - (short.y + short.height)).toBeLessThan(8);
});
