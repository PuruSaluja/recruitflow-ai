import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";

import type { ApiClient } from "./api/client";
import { ApiProvider } from "./api/context";
import { AppRoutes } from "./App";
import {
  fakeApi,
  makeApplication,
  makeCandidate,
  makeJob,
  makeRanked,
  page,
} from "./test/fakeApi";

const HEALTH = { "GET /health": () => ({ body: { status: "ok", version: "0.1.0", database: "ok" } }) };

function renderAt(path: string, client: ApiClient) {
  return render(
    <ApiProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <AppRoutes />
      </MemoryRouter>
    </ApiProvider>,
  );
}

describe("job list", () => {
  it("lists jobs with links to each and pages through them", async () => {
    const jobs = Array.from({ length: 20 }, (_, i) => makeJob({ id: i + 1, title: `Role ${i + 1}` }));
    const api = fakeApi({
      ...HEALTH,
      "GET /jobs": (url) => {
        const offset = Number(url.searchParams.get("offset"));
        return { body: page(offset ? [makeJob({ id: 21, title: "Role 21" })] : jobs, 21, 20, offset) };
      },
    });
    renderAt("/jobs", api.client);

    const link = await screen.findByRole("link", { name: "Role 1" });
    expect(link.getAttribute("href")).toBe("/jobs/1");
    expect(screen.getByText("1–20 of 21")).toBeTruthy();
    expect(api.calls("GET /jobs")[0]?.url.search).toBe("?limit=20&offset=0");

    await userEvent.click(screen.getByRole("button", { name: "Next" }));

    expect(await screen.findByRole("link", { name: "Role 21" })).toBeTruthy();
    expect(screen.getByText("21–21 of 21")).toBeTruthy();
    expect(api.calls("GET /jobs").at(-1)?.url.searchParams.get("offset")).toBe("20");
    expect((screen.getByRole("button", { name: "Next" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("sends the title search and the status filter to the API", async () => {
    const api = fakeApi({ ...HEALTH, "GET /jobs": () => ({ body: page([makeJob()]) }) });
    renderAt("/jobs", api.client);
    await screen.findByRole("link", { name: "Backend Engineer" });

    await userEvent.type(screen.getByRole("searchbox", { name: "Search job titles" }), "  data  ");
    await userEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => expect(api.calls("GET /jobs").at(-1)?.url.searchParams.get("q")).toBe("data"));

    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Status" }), "paused");
    await waitFor(() =>
      expect(api.calls("GET /jobs").at(-1)?.url.searchParams.get("status")).toBe("paused"),
    );
    expect(api.calls("GET /jobs").at(-1)?.url.searchParams.get("q")).toBe("data");
  });

  it("ignores a status the API does not have and an offset that is not a number", async () => {
    const api = fakeApi({ ...HEALTH, "GET /jobs": () => ({ body: page([makeJob()]) }) });
    renderAt("/jobs?status=archived&offset=-5", api.client);
    await screen.findByRole("link", { name: "Backend Engineer" });

    const params = api.calls("GET /jobs")[0]?.url.searchParams;
    expect(params?.get("status")).toBeNull();
    expect(params?.get("offset")).toBe("0");
  });

  it("says when nothing matches", async () => {
    const api = fakeApi({ ...HEALTH, "GET /jobs": () => ({ body: page([]) }) });
    renderAt("/jobs?q=zzz", api.client);

    expect(await screen.findByText("No jobs match.")).toBeTruthy();
  });

  it("shows the API's error and loads again on request", async () => {
    let failing = true;
    const api = fakeApi({
      ...HEALTH,
      "GET /jobs": () =>
        failing
          ? { status: 503, body: { detail: "Database unavailable", code: "db_unavailable" } }
          : { body: page([makeJob()]) },
    });
    renderAt("/jobs", api.client);

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("Database unavailable");

    failing = false;
    await userEvent.click(within(alert).getByRole("button", { name: "Try again" }));

    expect(await screen.findByRole("link", { name: "Backend Engineer" })).toBeTruthy();
  });

  it("is where the root address goes", async () => {
    const api = fakeApi({ ...HEALTH, "GET /jobs": () => ({ body: page([makeJob()]) }) });
    renderAt("/", api.client);

    expect(await screen.findByRole("heading", { name: "Jobs" })).toBeTruthy();
  });
});

describe("job detail", () => {
  const MODEL = "3f9a1c2b7d4e5f60718293a4b5c6d7e8";
  const ranking = [
    makeRanked(1, { id: 10, full_name: "Amara Okafor" }, 0.874),
    makeRanked(2, { id: 11, full_name: "Wei Chen", current_title: null }, 0.412, 1),
  ];

  it("shows the job and its ranked candidates, best first", async () => {
    const api = fakeApi({
      ...HEALTH,
      "GET /jobs/1": () => ({
        body: makeJob({ requirement_notes: ["Senior roles usually ask for 5+ years."] }),
      }),
      "GET /jobs/1/matches": () => ({ body: { ...page(ranking, 2, 25), job_id: 1, model_version: MODEL } }),
    });
    renderAt("/jobs/1", api.client);

    expect(await screen.findByRole("heading", { name: "Backend Engineer" })).toBeTruthy();
    expect(screen.getByRole("note").textContent).toContain("Senior roles usually ask for 5+ years.");
    expect(screen.getByText("120,000 – 150,000")).toBeTruthy();
    expect(screen.getByText("5 years")).toBeTruthy();

    const table = await screen.findByRole("table");
    const rows = within(table).getAllByRole("row").slice(1);
    expect(rows.map((row) => within(row).getAllByRole("cell")[0]?.textContent)).toEqual(["1", "2"]);
    expect(within(rows[0]!).getByRole("link", { name: "Amara Okafor" }).getAttribute("href")).toBe(
      "/candidates/10",
    );
    expect(rows[0]!.textContent).toContain("87%");
    expect(rows[1]!.textContent).toContain("41%");
    // Required skills held: 50% for the first, all of them for the second.
    expect(within(rows[0]!).getAllByRole("cell")[3]?.textContent).toBe("50%");
    expect(within(rows[1]!).getAllByRole("cell")[3]?.textContent).toBe("100%");
    expect(within(rows[1]!).getByRole("link", { name: "PDF" }).getAttribute("href")).toBe(
      "/api/v1/jobs/1/candidates/11/scorecard.pdf",
    );
    expect(screen.getByRole("link", { name: "Shortlist PDF" }).getAttribute("href")).toBe(
      "/api/v1/jobs/1/report.pdf",
    );
    expect(screen.getByText(/Matcher 3f9a1c2b7d4e\./)).toBeTruthy();
    expect(api.calls("GET /jobs/1/matches")[0]?.url.search).toBe("?limit=25&offset=0");
  });

  it("scores the pool when asked and then lists the new ranking", async () => {
    let scored = false;
    const api = fakeApi({
      ...HEALTH,
      "GET /jobs/1": () => ({ body: makeJob() }),
      "GET /jobs/1/matches": () => ({
        body: { ...page(scored ? ranking : [], scored ? 2 : 0, 25), job_id: 1, model_version: MODEL },
      }),
      "POST /jobs/1/score": () => {
        scored = true;
        return {
          body: { job_id: 1, model_version: MODEL, scored_count: 2, elapsed_ms: 41.3, results: ranking.slice(0, 1) },
        };
      },
    });
    renderAt("/jobs/1", api.client);

    expect(await screen.findByText(/No candidate has been scored/)).toBeTruthy();
    await userEvent.click(screen.getByRole("button", { name: "Score all candidates" }));

    expect(await screen.findByRole("link", { name: "Wei Chen" })).toBeTruthy();
    expect(screen.getByText("Scored 2 candidates in 41 ms.")).toBeTruthy();
    expect(api.calls("POST /jobs/1/score")[0]?.init?.body).toBe('{"limit":1}');
  });

  it("keeps the job on screen when the matcher cannot be loaded", async () => {
    const api = fakeApi({
      ...HEALTH,
      "GET /jobs/1": () => ({ body: makeJob() }),
      "GET /jobs/1/matches": () => ({
        status: 503,
        body: { detail: "The trained matcher is missing or cannot be loaded.", code: "matcher_unavailable" },
      }),
    });
    renderAt("/jobs/1", api.client);

    expect(await screen.findByRole("heading", { name: "Backend Engineer" })).toBeTruthy();
    expect((await screen.findByRole("alert")).textContent).toContain("trained matcher is missing");
  });

  it("shows a scoring failure beside the button", async () => {
    const api = fakeApi({
      ...HEALTH,
      "GET /jobs/1": () => ({ body: makeJob() }),
      "GET /jobs/1/matches": () => ({ body: { ...page([], 0, 25), job_id: 1, model_version: MODEL } }),
      "POST /jobs/1/score": () => ({ status: 409, body: { detail: "Scoring job 1 collided. Retry the request." } }),
    });
    renderAt("/jobs/1", api.client);
    await screen.findByText(/No candidate has been scored/);

    await userEvent.click(screen.getByRole("button", { name: "Score all candidates" }));

    expect((await screen.findByRole("alert")).textContent).toContain("collided");
    expect((screen.getByRole("button", { name: "Score all candidates" }) as HTMLButtonElement).disabled).toBe(
      false,
    );
  });

  it("says there is no such job for a 404 and for an id the API could not hold", async () => {
    const api = fakeApi({ ...HEALTH, "GET /jobs/7": () => ({ status: 404, body: { detail: "Job 7 not found" } }) });
    const { unmount } = renderAt("/jobs/7", api.client);
    expect(await screen.findByRole("heading", { name: "No such job" })).toBeTruthy();
    unmount();

    for (const bad of ["abc", "0", "2147483648", "1.5"]) {
      const view = renderAt(`/jobs/${bad}`, api.client);
      expect(await screen.findByRole("heading", { name: "No such job" })).toBeTruthy();
      view.unmount();
    }
    expect(api.fetchImpl.mock.calls.filter(([url]) => String(url).includes("/jobs/"))).toHaveLength(1);
  });
});

describe("candidate detail", () => {
  it("shows the candidate, their resume and their applications with job titles", async () => {
    const api = fakeApi({
      ...HEALTH,
      "GET /candidates/10": () => ({ body: makeCandidate({ phone: "+1 613 555 0100" }) }),
      "GET /applications": () => ({
        body: page([
          makeApplication({ id: 100, job_id: 1, status: "interview" }),
          makeApplication({ id: 101, job_id: 2, status: "rejected", source: null }),
        ]),
      }),
      "GET /jobs/1": () => ({ body: makeJob() }),
      "GET /jobs/2": () => ({ status: 404, body: { detail: "Job 2 not found" } }),
    });
    renderAt("/candidates/10", api.client);

    expect(await screen.findByRole("heading", { name: "Amara Okafor" })).toBeTruthy();
    expect(screen.getByText("+1 613 555 0100")).toBeTruthy();
    expect(screen.getByText("Six years building Python services on PostgreSQL.")).toBeTruthy();
    expect(screen.getByText("Kubernetes")).toBeTruthy();

    const first = await screen.findByRole("link", { name: "Backend Engineer" });
    expect(first.getAttribute("href")).toBe("/jobs/1");
    expect(screen.getByRole("link", { name: "Job 2" }).getAttribute("href")).toBe("/jobs/2");
    expect(screen.getByText("Interview")).toBeTruthy();
    expect(screen.getByText("Rejected")).toBeTruthy();
    expect(api.calls("GET /applications")[0]?.url.search).toBe("?candidate_id=10&limit=200");
  });

  it("says how many applications are not shown", async () => {
    const api = fakeApi({
      ...HEALTH,
      "GET /candidates/10": () => ({ body: makeCandidate() }),
      "GET /applications": () => ({ body: page([makeApplication()], 201, 200) }),
      "GET /jobs/1": () => ({ body: makeJob() }),
    });
    renderAt("/candidates/10", api.client);

    expect(await screen.findByText("Showing the latest 1 of 201.")).toBeTruthy();
  });

  it("says there is no such candidate for a 404", async () => {
    const api = fakeApi({
      ...HEALTH,
      "GET /candidates/9": () => ({ status: 404, body: { detail: "Candidate 9 not found" } }),
    });
    renderAt("/candidates/9", api.client);

    expect(await screen.findByRole("heading", { name: "No such candidate" })).toBeTruthy();
  });
});

describe("layout", () => {
  it("answers an unknown address with a not-found page", async () => {
    renderAt("/nowhere", fakeApi(HEALTH).client);

    expect(await screen.findByRole("heading", { name: "Page not found" })).toBeTruthy();
  });

  it("shows the API's health in the header", async () => {
    renderAt("/nowhere", fakeApi(HEALTH).client);

    expect(await screen.findByText("API 0.1.0, database ok")).toBeTruthy();
  });
});
