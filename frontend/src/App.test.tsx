import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";

import type { ApiClient } from "./api/client";
import { ApiProvider } from "./api/context";
import { AppRoutes } from "./App";
import {
  fakeApi,
  makeJob,
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
