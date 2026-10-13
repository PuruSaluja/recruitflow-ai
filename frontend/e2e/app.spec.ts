import AxeBuilder from "@axe-core/playwright";
import { type APIRequestContext, type Page, expect, test } from "@playwright/test";

const API = "/api/v1";
const RUN = `${Date.now().toString(36)}${Math.floor(Math.random() * 1e6).toString(36)}`;
const JOB_TITLE = `Analytics Engineer ${RUN}`;

type Created = { job: number; candidates: number[]; applicant: { id: number; name: string } };

const PEOPLE = [
  {
    name: "Imani Clarke",
    title: "Analytics Engineer",
    years: 6,
    seniority: "senior",
    skills: ["SQL", "dbt", "Python", "Looker"],
  },
  { name: "Tomás Reyes", title: "Data Analyst", years: 3, seniority: "mid", skills: ["SQL", "Excel"] },
  { name: "Mei Tanaka", title: "Graphic Designer", years: 4, seniority: "mid", skills: ["Figma"] },
];

async function ok<T>(response: Awaited<ReturnType<APIRequestContext["get"]>>, status = 200): Promise<T> {
  expect(response.status(), await response.text()).toBe(status);
  return (status === 204 ? undefined : await response.json()) as T;
}

async function seed(request: APIRequestContext): Promise<Created> {
  const job = await ok<{ id: number }>(
    await request.post(`${API}/jobs`, {
      data: {
        title: JOB_TITLE,
        department: "Data",
        location: "Toronto, ON",
        seniority: "senior",
        min_years_experience: 4,
        description: "Model the warehouse in dbt and own the metrics layer behind every dashboard.",
        required_skills: ["SQL", "dbt"],
        preferred_skills: ["Looker"],
      },
    }),
    201,
  );
  const candidates: number[] = [];
  for (const [index, person] of PEOPLE.entries()) {
    const created = await ok<{ id: number }>(
      await request.post(`${API}/candidates`, {
        data: {
          full_name: person.name,
          // As long as real addresses get: one unbreakable word on a phone.
          email: `imani.clarke.okonkwo.${RUN}.${index}@peopleoperations.example.com`,
          current_title: person.title,
          years_experience: person.years,
          seniority: person.seniority,
          resume_text: `${person.title} with ${person.years} years of experience using ${person.skills.join(", ")}.`,
          skills: person.skills,
        },
      }),
      201,
    );
    candidates.push(created.id);
  }
  const [first] = candidates;
  await ok(
    await request.post(`${API}/applications`, { data: { job_id: job.id, candidate_id: first, source: "referral" } }),
    201,
  );
  return { job: job.id, candidates, applicant: { id: first!, name: PEOPLE[0]!.name } };
}

async function remove(request: APIRequestContext, created: Created) {
  await ok(await request.delete(`${API}/jobs/${created.job}`), 204);
  for (const id of created.candidates) {
    await ok(await request.delete(`${API}/candidates/${id}`), 204);
  }
}

/** Serious and critical accessibility violations on the page as it stands. */
async function seriousViolations(page: Page) {
  const results = await new AxeBuilder({ page }).analyze();
  return results.violations
    .filter((violation) => violation.impact === "serious" || violation.impact === "critical")
    .map((violation) => `${violation.id}: ${violation.nodes.map((node) => node.target.join(" ")).join(", ")}`);
}

test.describe.configure({ mode: "serial" });

let created: Created;

test.beforeAll(async ({ request }) => {
  created = await seed(request);
});

test.afterAll(async ({ request }) => {
  if (created) await remove(request, created);
});

test("finds the job, scores its candidates and links working PDFs", async ({ page, request }) => {
  await page.goto("/jobs");
  await page.getByRole("searchbox", { name: "Search job titles" }).fill(RUN);
  await page.getByRole("button", { name: "Search" }).click();
  await page.getByRole("link", { name: JOB_TITLE }).click();

  await expect(page.getByRole("heading", { name: JOB_TITLE })).toBeVisible();
  await expect(page.getByText("No candidate has been scored")).toBeVisible();
  await expect(page.getByRole("link", { name: "Shortlist PDF" })).toHaveCount(0);

  await page.getByRole("button", { name: "Score all candidates" }).click();

  await expect(page.getByText(/^Scored \d+ candidates? in \d+ ms\.$/)).toBeVisible();
  const table = page.getByRole("table");
  for (const person of PEOPLE) {
    await expect(table.getByRole("link", { name: person.name, exact: true })).toBeVisible();
  }

  const shortlist = await page.getByRole("link", { name: "Shortlist PDF" }).getAttribute("href");
  const scorecard = await page
    .getByRole("link", { name: `Scorecard for ${created.applicant.name}` })
    .getAttribute("href");
  for (const href of [shortlist, scorecard]) {
    const response = await request.get(href!);
    expect(response.status()).toBe(200);
    expect(response.headers()["content-type"]).toBe("application/pdf");
    expect((await response.body()).subarray(0, 5).toString()).toBe("%PDF-");
  }
});

test("opens a candidate from the ranking, and a reload keeps the page", async ({ page }) => {
  await page.goto(`/jobs/${created.job}`);
  await page.getByRole("table").getByRole("link", { name: created.applicant.name, exact: true }).click();

  await expect(page).toHaveURL(new RegExp(`/candidates/${created.applicant.id}$`));
  await expect(page.getByRole("heading", { name: created.applicant.name })).toBeVisible();
  const application = page.getByRole("link", { name: JOB_TITLE });
  await expect(application).toHaveAttribute("href", `/jobs/${created.job}`);

  // The server answers a page address with the client, so a reload works.
  await page.reload();
  await expect(page.getByRole("heading", { name: created.applicant.name })).toBeVisible();
});

test("answers unknown records and addresses with not-found pages", async ({ page }) => {
  await page.goto("/jobs/2147483647");
  await expect(page.getByRole("heading", { name: "No such job" })).toBeVisible();

  await page.goto("/candidates/abc");
  await expect(page.getByRole("heading", { name: "No such candidate" })).toBeVisible();

  await page.goto("/nowhere");
  await expect(page.getByRole("heading", { name: "Page not found" })).toBeVisible();
});

for (const colorScheme of ["light", "dark"] as const) {
  test(`has no serious accessibility violations on any page, ${colorScheme}`, async ({
    page,
    request,
  }) => {
    // Contrast is checked against the colours each scheme actually draws.
    await page.emulateMedia({ colorScheme });
    // Scored here as well, so the ranking is on the page when this test runs alone.
    await ok(
      await request.post(`${API}/jobs/${created.job}/score`, { data: { candidate_ids: created.candidates } }),
    );

    await page.goto(`/jobs?q=${RUN}`);
    await expect(page.getByRole("link", { name: JOB_TITLE })).toBeVisible();
    expect(await seriousViolations(page)).toEqual([]);

    await page.goto(`/jobs/${created.job}`);
    await expect(page.getByRole("table")).toBeVisible();
    expect(await seriousViolations(page)).toEqual([]);

    await page.goto(`/candidates/${created.applicant.id}`);
    await expect(page.getByRole("link", { name: JOB_TITLE })).toBeVisible();
    expect(await seriousViolations(page)).toEqual([]);
  });
}

test("never scrolls the page sideways", async ({ page }) => {
  for (const path of [`/jobs`, `/jobs/${created.job}`, `/candidates/${created.applicant.id}`]) {
    await page.goto(path);
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow, path).toBeLessThanOrEqual(0);
  }
});
