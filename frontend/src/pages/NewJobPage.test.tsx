import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { fakeApi, makeJob, page } from "../test/fakeApi";
import { HEALTH, renderAt } from "../test/render";

const MATCHES = {
  "GET /jobs/42/matches": () => ({ body: { ...page([], 0, 25), job_id: 42, model_version: "v1" } }),
};

describe("new job", () => {
  it("is reached from the job list", async () => {
    const api = fakeApi({ ...HEALTH, "GET /jobs": () => ({ body: page([makeJob()]) }) });
    renderAt("/jobs", api.client);

    await userEvent.click(await screen.findByRole("link", { name: "New job" }));

    expect(await screen.findByRole("heading", { name: "New job" })).toBeTruthy();
  });

  it("creates the job from the form, leaving blanks to the API, and opens it", async () => {
    const api = fakeApi({
      ...HEALTH,
      ...MATCHES,
      "POST /jobs": () => ({ status: 201, body: makeJob({ id: 42, title: "Data Engineer" }) }),
      "GET /jobs/42": () => ({ body: makeJob({ id: 42, title: "Data Engineer" }) }),
    });
    renderAt("/jobs/new", api.client);

    await userEvent.type(await screen.findByLabelText("Title"), "Data Engineer");
    await userEvent.selectOptions(screen.getByLabelText("Seniority"), "senior");
    await userEvent.clear(screen.getByLabelText("Minimum years of experience"));
    await userEvent.type(screen.getByLabelText("Minimum years of experience"), "4");
    await userEvent.type(screen.getByLabelText("Required skills"), " Python, SQL ,, dbt ");
    await userEvent.type(screen.getByLabelText("Salary from"), "95000.50");
    await userEvent.type(screen.getByLabelText("Description"), "Build and run the nightly loads.");
    await userEvent.click(screen.getByRole("button", { name: "Create job" }));

    expect(await screen.findByRole("heading", { name: "Data Engineer" })).toBeTruthy();
    const [sent] = api.calls("POST /jobs");
    expect(JSON.parse(String(sent!.init!.body))).toEqual({
      title: "Data Engineer",
      description: "Build and run the nightly loads.",
      employment_type: "full_time",
      seniority: "senior",
      education_required: "bachelor",
      min_years_experience: 4,
      required_skills: ["Python", "SQL", "dbt"],
      preferred_skills: [],
      // The number box gives its value as text, trailing zero dropped; it goes as
      // that text, never through a float.
      salary_min: "95000.5",
      status: "open",
    });
  });

  it("shows the API's refusal and keeps what was typed", async () => {
    const api = fakeApi({
      ...HEALTH,
      "POST /jobs": () => ({
        status: 422,
        body: { detail: [{ loc: ["body"], msg: "Value error, salary_max must be greater than or equal to salary_min" }] },
      }),
    });
    renderAt("/jobs/new", api.client);

    await userEvent.type(await screen.findByLabelText("Title"), "Data Engineer");
    await userEvent.type(screen.getByLabelText("Salary from"), "200000");
    await userEvent.type(screen.getByLabelText("Salary to"), "100000");
    await userEvent.type(screen.getByLabelText("Description"), "Build and run the nightly loads.");
    await userEvent.click(screen.getByRole("button", { name: "Create job" }));

    expect((await screen.findByRole("alert")).textContent).toContain(
      "salary_max must be greater than or equal to salary_min",
    );
    expect((screen.getByLabelText("Title") as HTMLInputElement).value).toBe("Data Engineer");
    expect(screen.getByRole("heading", { name: "New job" })).toBeTruthy();
    expect((screen.getByRole("button", { name: "Create job" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("sends one request however fast the button is pressed", async () => {
    let answer: (value: { status: number; body: unknown }) => void = () => {};
    const api = fakeApi({
      ...HEALTH,
      ...MATCHES,
      "POST /jobs": () => new Promise((resolve) => (answer = resolve)),
      "GET /jobs/42": () => ({ body: makeJob({ id: 42, title: "Data Engineer" }) }),
    });
    renderAt("/jobs/new", api.client);
    await userEvent.type(await screen.findByLabelText("Title"), "Data Engineer");
    await userEvent.type(screen.getByLabelText("Description"), "Build and run the nightly loads.");

    await userEvent.dblClick(screen.getByRole("button", { name: "Create job" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Saving…" })).toBeTruthy());
    answer({ status: 201, body: makeJob({ id: 42, title: "Data Engineer" }) });

    expect(await screen.findByRole("heading", { name: "Data Engineer" })).toBeTruthy();
    expect(api.calls("POST /jobs")).toHaveLength(1);
  });

  it("names every field for assistive technology, with hints read after the name", async () => {
    renderAt("/jobs/new", fakeApi({ ...HEALTH }).client);

    const skills = await screen.findByRole("textbox", { name: "Required skills" });
    expect(skills.getAttribute("aria-describedby")).toBeTruthy();
    expect(document.getElementById(skills.getAttribute("aria-describedby")!)?.textContent).toContain(
      "Separate them with commas",
    );
    expect(screen.getByRole("textbox", { name: "Title" }).getAttribute("aria-describedby")).toBeNull();
  });
});
