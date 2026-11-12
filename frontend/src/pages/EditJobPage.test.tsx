import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { Job } from "../api/client";
import { fakeApi, makeJob, page } from "../test/fakeApi";
import { HEALTH, renderAt } from "../test/render";

const STORED = makeJob({ id: 42, title: "Data Engineer" });

/** An API holding one job, which a PATCH changes as the real one would. */
function apiWith(job: Job = STORED) {
  let stored = job;
  return fakeApi({
    ...HEALTH,
    "GET /jobs/42": () => ({ body: stored }),
    "GET /jobs/42/matches": () => ({ body: { ...page([], 0, 25), job_id: 42, model_version: "v1" } }),
    "PATCH /jobs/42": (_url, init) => {
      stored = { ...stored, ...(JSON.parse(String(init.body)) as Partial<Job>) };
      return { body: stored };
    },
  });
}

function value(name: string): string {
  return (screen.getByLabelText(name) as HTMLInputElement).value;
}

function sent(api: ReturnType<typeof apiWith>): unknown[] {
  return api.calls("PATCH /jobs/42").map((call) => JSON.parse(String(call.init!.body)));
}

describe("editing a job", () => {
  it("is reached from the job's page and shows the job as stored", async () => {
    renderAt("/jobs/42", apiWith().client);

    await userEvent.click(await screen.findByRole("link", { name: "Edit" }));

    expect(await screen.findByRole("heading", { name: "Edit job" })).toBeTruthy();
    expect(value("Title")).toBe("Data Engineer");
    expect(value("Department")).toBe("Engineering");
    expect(value("Seniority")).toBe("senior");
    expect(value("Minimum years of experience")).toBe("5");
    expect(value("Required skills")).toBe("Python, PostgreSQL");
    expect(value("Preferred skills")).toBe("Docker");
    expect(value("Salary from")).toBe("120000.00");
    expect(value("Salary to")).toBe("150000.00");
    expect(value("Status")).toBe("open");
    expect(value("Description")).toBe(STORED.description);
  });

  it("offers every status, where a new job starts open or as a draft", async () => {
    renderAt("/jobs/42/edit", apiWith().client);

    const status = (await screen.findByLabelText("Status")) as HTMLSelectElement;

    expect([...status.options].map((option) => option.value)).toEqual(["open", "paused", "draft", "closed"]);
  });

  it("sends only what was changed, a cleared box as null, and opens the job", async () => {
    const api = apiWith();
    renderAt("/jobs/42/edit", api.client);

    await userEvent.type(await screen.findByLabelText("Title"), " II");
    await userEvent.clear(screen.getByLabelText("Department"));
    await userEvent.clear(screen.getByLabelText("Salary to"));
    await userEvent.type(screen.getByLabelText("Required skills"), ", dbt");
    await userEvent.selectOptions(screen.getByLabelText("Status"), "paused");
    await userEvent.click(screen.getByRole("button", { name: "Save changes" }));

    expect(await screen.findByRole("heading", { name: "Data Engineer II" })).toBeTruthy();
    expect(sent(api)).toEqual([
      {
        title: "Data Engineer II",
        department: null,
        required_skills: ["Python", "PostgreSQL", "dbt"],
        salary_max: null,
        status: "paused",
      },
    ]);
    expect(screen.getByText("Paused")).toBeTruthy();
  });

  it("asks nothing of the API when nothing was changed", async () => {
    const api = apiWith();
    renderAt("/jobs/42/edit", api.client);

    await userEvent.click(await screen.findByRole("button", { name: "Save changes" }));

    expect(await screen.findByRole("heading", { name: "Data Engineer" })).toBeTruthy();
    expect(sent(api)).toEqual([]);
  });

  it("leaves alone a field the form cannot spell back, unless it is edited", async () => {
    // A comma inside one skill: split at commas, the box would make two of it.
    const api = apiWith(makeJob({ id: 42, required_skills: ["C, C++"], salary_min: null, location: null }));
    renderAt("/jobs/42/edit", api.client);

    await userEvent.type(await screen.findByLabelText("Location"), "Remote");
    await userEvent.click(screen.getByRole("button", { name: "Save changes" }));

    await screen.findByRole("heading", { name: "Backend Engineer" });
    expect(sent(api)).toEqual([{ location: "Remote" }]);
  });

  it("shows the API's refusal and keeps what was typed", async () => {
    const api = fakeApi({
      ...HEALTH,
      "GET /jobs/42": () => ({ body: STORED }),
      "PATCH /jobs/42": () => ({
        status: 422,
        body: { detail: "salary_max must be greater than or equal to salary_min" },
      }),
    });
    renderAt("/jobs/42/edit", api.client);

    await userEvent.clear(await screen.findByLabelText("Salary from"));
    await userEvent.type(screen.getByLabelText("Salary from"), "200000");
    await userEvent.click(screen.getByRole("button", { name: "Save changes" }));

    expect((await screen.findByRole("alert")).textContent).toContain(
      "salary_max must be greater than or equal to salary_min",
    );
    expect(value("Salary from")).toBe("200000");
    expect(screen.getByRole("heading", { name: "Edit job" })).toBeTruthy();
    expect((screen.getByRole("button", { name: "Save changes" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("goes back to the job without saving", async () => {
    const api = apiWith();
    renderAt("/jobs/42/edit", api.client);

    await userEvent.type(await screen.findByLabelText("Title"), " II");
    await userEvent.click(screen.getByRole("link", { name: "Cancel" }));

    expect(await screen.findByRole("heading", { name: "Data Engineer" })).toBeTruthy();
    expect(sent(api)).toEqual([]);
  });

  it("says there is no such job for a 404 and for an id the API could not hold", async () => {
    const api = fakeApi({ ...HEALTH, "GET /jobs/7": () => ({ status: 404, body: { detail: "Job 7 not found" } }) });
    const first = renderAt("/jobs/7/edit", api.client);
    expect(await screen.findByRole("heading", { name: "No such job" })).toBeTruthy();
    first.unmount();

    renderAt("/jobs/99999999999/edit", api.client);
    expect(await screen.findByRole("heading", { name: "No such job" })).toBeTruthy();
    expect(api.calls("GET /jobs/99999999999")).toEqual([]);
  });
});
