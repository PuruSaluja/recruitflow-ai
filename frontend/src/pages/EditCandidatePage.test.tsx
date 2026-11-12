import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { Candidate } from "../api/client";
import { fakeApi, makeCandidate, page } from "../test/fakeApi";
import { HEALTH, renderAt } from "../test/render";

const STORED = makeCandidate({ id: 77, full_name: "Ana Diaz", email: "ana@example.com" });

/** An API holding one candidate, whom a PATCH changes as the real one would. */
function apiWith(candidate: Candidate = STORED) {
  let stored = candidate;
  return fakeApi({
    ...HEALTH,
    "GET /candidates/77": () => ({ body: stored }),
    "GET /applications": () => ({ body: page([]) }),
    "PATCH /candidates/77": (_url, init) => {
      stored = { ...stored, ...(JSON.parse(String(init.body)) as Partial<Candidate>) };
      return { body: stored };
    },
  });
}

function value(name: string): string {
  return (screen.getByLabelText(name) as HTMLInputElement).value;
}

function sent(api: ReturnType<typeof apiWith>): unknown[] {
  return api.calls("PATCH /candidates/77").map((call) => JSON.parse(String(call.init!.body)));
}

describe("editing a candidate", () => {
  it("is reached from the candidate's page and shows them as stored", async () => {
    renderAt("/candidates/77", apiWith().client);

    await userEvent.click(await screen.findByRole("link", { name: "Edit" }));

    expect(await screen.findByRole("heading", { name: "Edit candidate" })).toBeTruthy();
    expect(value("Full name")).toBe("Ana Diaz");
    expect(value("Email")).toBe("ana@example.com");
    expect(value("Phone")).toBe("");
    expect(value("Location")).toBe("Ottawa, ON");
    expect(value("Current title")).toBe("Software Engineer");
    expect(value("Years of experience")).toBe("6");
    expect(value("Seniority")).toBe("senior");
    expect(value("Education")).toBe("master");
    expect(value("Skills")).toBe("Python, PostgreSQL, Kubernetes");
    expect(value("Resume")).toBe(STORED.resume_text);
    // The file box belongs to adding a candidate; their page replaces a resume from a file.
    expect(screen.queryByLabelText("Resume file")).toBeNull();
  });

  it("sends only what was changed, a cleared box as null, and opens the candidate", async () => {
    const api = apiWith();
    renderAt("/candidates/77/edit", api.client);

    await userEvent.type(await screen.findByLabelText("Full name"), "-Reyes");
    await userEvent.type(screen.getByLabelText("Phone"), " 613 555 0100 ");
    await userEvent.clear(screen.getByLabelText("Location"));
    await userEvent.clear(screen.getByLabelText("Years of experience"));
    await userEvent.type(screen.getByLabelText("Years of experience"), "7");
    await userEvent.type(screen.getByLabelText("Skills"), ", Rust");
    await userEvent.click(screen.getByRole("button", { name: "Save changes" }));

    expect(await screen.findByRole("heading", { name: "Ana Diaz-Reyes" })).toBeTruthy();
    expect(sent(api)).toEqual([
      {
        full_name: "Ana Diaz-Reyes",
        phone: "613 555 0100",
        location: null,
        years_experience: 7,
        skills: ["Python", "PostgreSQL", "Kubernetes", "Rust"],
      },
    ]);
  });

  it("sends the resume with emptied skills, so the API takes them from it", async () => {
    const api = apiWith();
    renderAt("/candidates/77/edit", api.client);

    await userEvent.clear(await screen.findByLabelText("Skills"));
    await userEvent.click(screen.getByRole("button", { name: "Save changes" }));

    await screen.findByRole("heading", { name: "Ana Diaz" });
    expect(sent(api)).toEqual([{ skills: [], resume_text: STORED.resume_text }]);
  });

  it("sends the resume for a candidate stored without skills, whatever else was changed", async () => {
    const api = apiWith({ ...STORED, skills: [] });
    renderAt("/candidates/77/edit", api.client);

    await userEvent.click(await screen.findByRole("button", { name: "Save changes" }));

    await screen.findByRole("heading", { name: "Ana Diaz" });
    expect(sent(api)).toEqual([{ resume_text: STORED.resume_text }]);
  });

  it("asks nothing of the API when nothing was changed", async () => {
    const api = apiWith();
    renderAt("/candidates/77/edit", api.client);

    await userEvent.click(await screen.findByRole("button", { name: "Save changes" }));

    expect(await screen.findByRole("heading", { name: "Ana Diaz" })).toBeTruthy();
    expect(sent(api)).toEqual([]);
  });

  it("shows a duplicate email as the API words it and keeps what was typed", async () => {
    const api = fakeApi({
      ...HEALTH,
      "GET /candidates/77": () => ({ body: STORED }),
      "PATCH /candidates/77": () => ({
        status: 409,
        body: { detail: "A candidate with this email already exists", code: "conflict" },
      }),
    });
    renderAt("/candidates/77/edit", api.client);

    await userEvent.clear(await screen.findByLabelText("Email"));
    await userEvent.type(screen.getByLabelText("Email"), "taken@example.com");
    await userEvent.click(screen.getByRole("button", { name: "Save changes" }));

    expect((await screen.findByRole("alert")).textContent).toContain(
      "A candidate with this email already exists",
    );
    expect(value("Email")).toBe("taken@example.com");
    expect(screen.getByRole("heading", { name: "Edit candidate" })).toBeTruthy();
    expect((screen.getByRole("button", { name: "Save changes" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("goes back to the candidate without saving", async () => {
    const api = apiWith();
    renderAt("/candidates/77/edit", api.client);

    await userEvent.type(await screen.findByLabelText("Full name"), "-Reyes");
    await userEvent.click(screen.getByRole("link", { name: "Cancel" }));

    expect(await screen.findByRole("heading", { name: "Ana Diaz" })).toBeTruthy();
    expect(sent(api)).toEqual([]);
  });

  it("says there is no such candidate for a 404 and for an id the API could not hold", async () => {
    const api = fakeApi({
      ...HEALTH,
      "GET /candidates/7": () => ({ status: 404, body: { detail: "Candidate 7 not found" } }),
    });
    const first = renderAt("/candidates/7/edit", api.client);
    expect(await screen.findByRole("heading", { name: "No such candidate" })).toBeTruthy();
    first.unmount();

    renderAt("/candidates/99999999999/edit", api.client);
    expect(await screen.findByRole("heading", { name: "No such candidate" })).toBeTruthy();
    expect(api.calls("GET /candidates/99999999999")).toEqual([]);
  });
});
