import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { Application } from "../api/client";
import { fakeApi, makeApplication, makeCandidate, makeJob, page } from "../test/fakeApi";
import { HEALTH, renderAt } from "../test/render";

const JOBS = [makeJob({ id: 1, title: "Backend Engineer" }), makeJob({ id: 2, title: "Data Engineer" })];

/** A candidate whose applications live in `applications`, which the handlers change. */
function candidateApi(applications: Application[], extra = {}) {
  return fakeApi({
    ...HEALTH,
    "GET /candidates/10": () => ({ body: makeCandidate() }),
    "GET /applications": () => ({ body: page(applications) }),
    "GET /jobs": (url) => {
      if (url.searchParams.get("status") === "open") return { body: page(JOBS) };
      const ids = url.searchParams.getAll("id").map(Number);
      return { body: page(JOBS.filter((job) => ids.includes(job.id))) };
    },
    ...extra,
  });
}

function stageCell(title: string): HTMLElement {
  const row = screen.getByRole("link", { name: title }).closest("tr")!;
  return within(row).getAllByRole("cell")[1]!;
}

/** The stage an application's row shows, not the moves its menu offers. */
function stageOf(title: string): string | null {
  return stageCell(title).querySelector(".stage > span")!.textContent;
}

describe("moving an application", () => {
  it("offers the stages the API lists, moves to the one chosen and shows it", async () => {
    const applications = [makeApplication({ id: 100, job_id: 1, status: "screening", next_statuses: ["interview", "rejected", "withdrawn"] })];
    const api = candidateApi(applications, {
      "PATCH /applications/100": (_url: URL, init: RequestInit) => {
        const { status } = JSON.parse(String(init.body)) as { status: Application["status"] };
        applications[0] = makeApplication({ id: 100, job_id: 1, status, next_statuses: ["offer", "rejected", "withdrawn"] });
        return { body: applications[0] };
      },
    });
    renderAt("/candidates/10", api.client);

    const select = await screen.findByRole("combobox", { name: "Next stage for Backend Engineer" });
    expect([...(select as HTMLSelectElement).options].map((option) => option.value)).toEqual([
      "",
      "interview",
      "rejected",
      "withdrawn",
    ]);
    const move = screen.getByRole("button", { name: "Move the application for Backend Engineer" });
    expect((move as HTMLButtonElement).disabled).toBe(true);

    await userEvent.selectOptions(select, "interview");
    await userEvent.click(move);

    await waitFor(() => expect(stageOf("Backend Engineer")).toBe("Interview"));
    expect(JSON.parse(String(api.calls("PATCH /applications/100")[0]!.init!.body))).toEqual({ status: "interview" });
  });

  it("offers no move for a finished application, and a reopen only where the API lists one", async () => {
    const api = candidateApi([
      makeApplication({ id: 100, job_id: 1, status: "hired", next_statuses: [], reopenable: false }),
      makeApplication({ id: 101, job_id: 2, status: "rejected", next_statuses: [], reopenable: true }),
    ]);
    renderAt("/candidates/10", api.client);

    await screen.findByRole("link", { name: "Backend Engineer" });
    expect(within(stageCell("Backend Engineer")).queryByRole("combobox")).toBeNull();
    expect(within(stageCell("Backend Engineer")).queryByRole("button")).toBeNull();
    expect(within(stageCell("Data Engineer")).queryByRole("combobox")).toBeNull();
    expect(screen.getByRole("button", { name: "Reopen the application for Data Engineer" })).toBeTruthy();
  });

  it("reopens a rejected application", async () => {
    const applications = [makeApplication({ id: 101, job_id: 2, status: "rejected", next_statuses: [], reopenable: true })];
    const api = candidateApi(applications, {
      "POST /applications/101/reopen": () => {
        applications[0] = makeApplication({ id: 101, job_id: 2, status: "applied", next_statuses: ["screening", "rejected", "withdrawn"] });
        return { body: applications[0] };
      },
    });
    renderAt("/candidates/10", api.client);

    await userEvent.click(await screen.findByRole("button", { name: "Reopen the application for Data Engineer" }));

    await waitFor(() => expect(stageOf("Data Engineer")).toBe("Applied"));
    expect(screen.queryByRole("button", { name: /^Reopen/ })).toBeNull();
  });

  it("shows a refused move in its row, as the API words it", async () => {
    const api = candidateApi(
      [makeApplication({ id: 101, job_id: 2, status: "rejected", next_statuses: [], reopenable: true })],
      {
        "POST /applications/101/reopen": () => ({
          status: 409,
          body: { detail: "Job 2 is closed; reopen the job first.", code: "conflict" },
        }),
      },
    );
    renderAt("/candidates/10", api.client);

    await userEvent.click(await screen.findByRole("button", { name: "Reopen the application for Data Engineer" }));

    expect(within(stageCell("Data Engineer")).getByRole("alert").textContent).toContain(
      "Job 2 is closed; reopen the job first.",
    );
  });
});

describe("applying to a job", () => {
  it("reads open jobs only when asked, leaves out those applied to, and applies", async () => {
    const applications = [makeApplication({ id: 100, job_id: 1 })];
    const api = candidateApi(applications, {
      "POST /applications": (_url: URL, init: RequestInit) => {
        const body = JSON.parse(String(init.body)) as { job_id: number; source?: string };
        const created = makeApplication({ id: 102, job_id: body.job_id, status: "applied", source: body.source ?? null });
        applications.unshift(created);
        return { status: 201, body: created };
      },
    });
    renderAt("/candidates/10", api.client);
    await screen.findByRole("link", { name: "Backend Engineer" });
    expect(api.calls("GET /jobs").some(({ url }) => url.searchParams.get("status") === "open")).toBe(false);

    await userEvent.click(screen.getByRole("button", { name: "Apply to a job" }));

    const job = await screen.findByRole("combobox", { name: "Job" });
    expect([...(job as HTMLSelectElement).options].map((option) => option.textContent)).toEqual([
      "Choose an open job",
      "Data Engineer",
    ]);
    await userEvent.selectOptions(job, "2");
    await userEvent.type(screen.getByLabelText("Source"), " referral ");
    await userEvent.click(screen.getByRole("button", { name: "Apply" }));

    expect(await screen.findByRole("link", { name: "Data Engineer" })).toBeTruthy();
    expect(screen.queryByRole("combobox", { name: "Job" })).toBeNull();
    expect(JSON.parse(String(api.calls("POST /applications")[0]!.init!.body))).toEqual({
      job_id: 2,
      candidate_id: 10,
      source: "referral",
    });
  });

  it("says so when every open job has been applied to", async () => {
    const api = candidateApi([makeApplication({ id: 100, job_id: 1 }), makeApplication({ id: 101, job_id: 2 })]);
    renderAt("/candidates/10", api.client);

    await userEvent.click(await screen.findByRole("button", { name: "Apply to a job" }));

    expect(await screen.findByText("There is no open job this candidate has not applied to.")).toBeTruthy();
  });
});

describe("replacing the resume", () => {
  it("uploads the file and shows the stored text and skills", async () => {
    let candidate = makeCandidate({ resume_text: "Old resume text, long enough.", skills: [] });
    const api = fakeApi({
      ...HEALTH,
      "GET /candidates/10": () => ({ body: candidate }),
      "GET /applications": () => ({ body: page([]) }),
      "PUT /candidates/10/resume": () => {
        candidate = makeCandidate({ resume_text: "Python and SQL for nine years.", skills: ["Python", "SQL"] });
        return { body: candidate };
      },
    });
    renderAt("/candidates/10", api.client);
    const file = new File(["Python and SQL for nine years."], "new.txt", { type: "text/plain" });

    await userEvent.upload(await screen.findByLabelText("Replace with a resume file"), file);

    expect(await screen.findByText("Python and SQL for nine years.")).toBeTruthy();
    expect(screen.getByText("SQL")).toBeTruthy();
    const sent = api.calls("PUT /candidates/10/resume")[0]!.init!.body as FormData;
    expect((sent.get("file") as File).name).toBe("new.txt");
  });

  it("shows why a file was refused and keeps the stored resume", async () => {
    const api = fakeApi({
      ...HEALTH,
      "GET /candidates/10": () => ({ body: makeCandidate() }),
      "GET /applications": () => ({ body: page([]) }),
      "PUT /candidates/10/resume": () => ({
        status: 400,
        body: { detail: "Word documents cannot be read; export the resume as a PDF." },
      }),
    });
    renderAt("/candidates/10", api.client);

    await userEvent.upload(
      await screen.findByLabelText("Replace with a resume file"),
      new File(["x"], "cv.docx", { type: "text/plain" }),
    );

    expect((await screen.findByRole("alert")).textContent).toContain("Word documents cannot be read");
    expect(screen.getByText("Six years building Python services on PostgreSQL.")).toBeTruthy();
  });
});
