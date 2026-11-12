import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { fakeApi, makeJob, page } from "../test/fakeApi";
import { HEALTH, renderAt } from "../test/render";

const MATCHES = {
  "GET /jobs/42/matches": () => ({ body: { ...page([], 0, 25), job_id: 42, model_version: "v1" } }),
};

describe("closing a job", () => {
  it("closes the job when asked and shows it closed, with nothing left to close", async () => {
    let status: "open" | "closed" = "open";
    const api = fakeApi({
      ...HEALTH,
      ...MATCHES,
      "GET /jobs/42": () => ({ body: makeJob({ id: 42, status }) }),
      "POST /jobs/42/close": () => {
        status = "closed";
        return { body: makeJob({ id: 42, status }) };
      },
    });
    renderAt("/jobs/42", api.client);
    expect((await screen.findByText("Open")).className).toContain("status-open");

    await userEvent.click(screen.getByRole("button", { name: "Close job" }));

    expect((await screen.findByText("Closed")).className).toContain("status-closed");
    expect(screen.queryByRole("button", { name: "Close job" })).toBeNull();
    expect(api.calls("POST /jobs/42/close")).toHaveLength(1);
    // Still editable, which is how it is opened again.
    expect(screen.getByRole("link", { name: "Edit" }).getAttribute("href")).toBe("/jobs/42/edit");
  });

  it("offers to close a paused job and a draft as well", async () => {
    for (const status of ["paused", "draft"] as const) {
      const api = fakeApi({ ...HEALTH, ...MATCHES, "GET /jobs/42": () => ({ body: makeJob({ id: 42, status }) }) });
      const shown = renderAt("/jobs/42", api.client);

      expect(await screen.findByRole("button", { name: "Close job" })).toBeTruthy();
      shown.unmount();
    }
  });

  it("offers no close for a job already closed", async () => {
    const api = fakeApi({
      ...HEALTH,
      ...MATCHES,
      "GET /jobs/42": () => ({ body: makeJob({ id: 42, status: "closed" }) }),
    });
    renderAt("/jobs/42", api.client);

    await screen.findByRole("heading", { name: "Backend Engineer" });

    expect(screen.queryByRole("button", { name: "Close job" })).toBeNull();
  });

  it("shows why the job could not be closed and leaves it as it was", async () => {
    const api = fakeApi({
      ...HEALTH,
      ...MATCHES,
      "GET /jobs/42": () => ({ body: makeJob({ id: 42 }) }),
      "POST /jobs/42/close": () => ({
        status: 503,
        body: { detail: "The database is unavailable", code: "db_unavailable" },
      }),
    });
    renderAt("/jobs/42", api.client);

    await userEvent.click(await screen.findByRole("button", { name: "Close job" }));

    expect((await screen.findByRole("alert")).textContent).toContain("The database is unavailable");
    expect(screen.getByText("Open")).toBeTruthy();
    await waitFor(() =>
      expect((screen.getByRole("button", { name: "Close job" }) as HTMLButtonElement).disabled).toBe(false),
    );
  });
});
