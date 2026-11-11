import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { fakeApi, makeCandidate, page } from "../test/fakeApi";
import { HEALTH, renderAt } from "../test/render";

const RESUME = "Five years of Python and PostgreSQL; deployed with Docker on AWS.";
const OPENED = {
  "GET /candidates/77": () => ({ body: makeCandidate({ id: 77, full_name: "Ana Diaz" }) }),
  "GET /applications": () => ({ body: page([]) }),
};

function resumeFile(text = RESUME, name = "cv.txt") {
  return new File([text], name, { type: "text/plain" });
}

describe("new candidate", () => {
  it("is reached from the top bar", async () => {
    const api = fakeApi({ ...HEALTH, "GET /jobs": () => ({ body: page([]) }) });
    renderAt("/jobs", api.client);

    await userEvent.click(screen.getByRole("link", { name: "Add candidate" }));

    expect(await screen.findByRole("heading", { name: "Add a candidate" })).toBeTruthy();
  });

  it("reads a resume file into the form, then saves the candidate and opens them", async () => {
    const api = fakeApi({
      ...HEALTH,
      ...OPENED,
      "POST /candidates/resume-preview": () => ({
        body: {
          filename: "cv.txt",
          characters: RESUME.length,
          words: 11,
          text: RESUME,
          skills: ["Python", "PostgreSQL", "Docker", "AWS"],
        },
      }),
      "POST /candidates": () => ({ status: 201, body: makeCandidate({ id: 77, full_name: "Ana Diaz" }) }),
    });
    renderAt("/candidates/new", api.client);

    await userEvent.upload(await screen.findByLabelText("Resume file"), resumeFile());

    expect((await screen.findByText(/^Read cv\.txt\./)).textContent).toBe(
      "Read cv.txt. It names Python, PostgreSQL, Docker, AWS.",
    );
    expect((screen.getByLabelText("Resume") as HTMLTextAreaElement).value).toBe(RESUME);
    expect((screen.getByLabelText("Skills") as HTMLInputElement).value).toBe("Python, PostgreSQL, Docker, AWS");
    const uploaded = api.calls("POST /candidates/resume-preview")[0]!.init!.body as FormData;
    expect((uploaded.get("file") as File).name).toBe("cv.txt");

    await userEvent.type(screen.getByLabelText("Full name"), "Ana Diaz");
    await userEvent.type(screen.getByLabelText("Email"), " ana@example.com ");
    await userEvent.click(screen.getByRole("button", { name: "Add candidate" }));

    expect(await screen.findByRole("heading", { name: "Ana Diaz" })).toBeTruthy();
    expect(JSON.parse(String(api.calls("POST /candidates")[0]!.init!.body))).toEqual({
      full_name: "Ana Diaz",
      email: "ana@example.com",
      years_experience: 0,
      seniority: "mid",
      education_level: "bachelor",
      resume_text: RESUME,
      skills: ["Python", "PostgreSQL", "Docker", "AWS"],
    });
  });

  it("keeps skills already typed when a resume is read", async () => {
    const api = fakeApi({
      ...HEALTH,
      "POST /candidates/resume-preview": () => ({
        body: { filename: "cv.txt", characters: 10, words: 2, text: RESUME, skills: ["Python"] },
      }),
    });
    renderAt("/candidates/new", api.client);

    await userEvent.type(await screen.findByLabelText("Skills"), "Rust");
    await userEvent.upload(screen.getByLabelText("Resume file"), resumeFile());

    await screen.findByText(/^Read cv\.txt\./);
    expect((screen.getByLabelText("Skills") as HTMLInputElement).value).toBe("Rust");
  });

  it("leaves empty skills out, so the API takes them from the pasted resume", async () => {
    const api = fakeApi({
      ...HEALTH,
      ...OPENED,
      "POST /candidates": () => ({ status: 201, body: makeCandidate({ id: 77, full_name: "Ana Diaz" }) }),
    });
    renderAt("/candidates/new", api.client);

    await userEvent.type(await screen.findByLabelText("Full name"), "Ana Diaz");
    await userEvent.type(screen.getByLabelText("Email"), "ana@example.com");
    await userEvent.type(screen.getByLabelText("Resume"), RESUME);
    await userEvent.click(screen.getByRole("button", { name: "Add candidate" }));

    await screen.findByRole("heading", { name: "Ana Diaz" });
    expect(JSON.parse(String(api.calls("POST /candidates")[0]!.init!.body))).not.toHaveProperty("skills");
  });

  it("shows why a resume file could not be read, and the form stays usable", async () => {
    const api = fakeApi({
      ...HEALTH,
      "POST /candidates/resume-preview": () => ({
        status: 400,
        body: { detail: "This PDF is encrypted. Remove the password and upload it again." },
      }),
    });
    renderAt("/candidates/new", api.client);

    await userEvent.upload(await screen.findByLabelText("Resume file"), resumeFile("x", "locked.pdf"));

    expect((await screen.findByRole("alert")).textContent).toContain("This PDF is encrypted.");
    await waitFor(() =>
      expect((screen.getByRole("button", { name: "Add candidate" }) as HTMLButtonElement).disabled).toBe(false),
    );
  });

  it("shows a duplicate email as the API words it", async () => {
    const api = fakeApi({
      ...HEALTH,
      "POST /candidates": () => ({
        status: 409,
        body: { detail: "A candidate with this email already exists", code: "conflict" },
      }),
    });
    renderAt("/candidates/new", api.client);

    await userEvent.type(await screen.findByLabelText("Full name"), "Ana Diaz");
    await userEvent.type(screen.getByLabelText("Email"), "ana@example.com");
    await userEvent.type(screen.getByLabelText("Resume"), RESUME);
    await userEvent.click(screen.getByRole("button", { name: "Add candidate" }));

    expect((await screen.findByRole("alert")).textContent).toContain(
      "A candidate with this email already exists",
    );
    expect(screen.getByRole("heading", { name: "Add a candidate" })).toBeTruthy();
  });
});
