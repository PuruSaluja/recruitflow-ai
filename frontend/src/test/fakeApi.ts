/**
 * A fake `fetch` answering the client's requests from a table of handlers, so
 * page tests run the real client, routes and components without a server.
 */
import { vi } from "vitest";

import type { Application, Candidate, Job, RankedCandidate } from "../api/client";
import { createClient } from "../api/client";

type Answer = { status?: number; body: unknown };
type Handler = (url: URL, init: RequestInit) => Answer | Promise<Answer>;

/** Handlers keyed by "METHOD /path" (path without the /api/v1 prefix or query). */
export function fakeApi(handlers: Record<string, Handler>) {
  const fetchImpl = vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = new URL(String(input), "http://localhost");
    const path = url.pathname.replace(/^\/api\/v1/, "");
    const key = `${init.method ?? "GET"} ${path}`;
    const handler = handlers[key];
    if (!handler) {
      return new Response(JSON.stringify({ detail: `No fake for ${key}` }), { status: 599 });
    }
    const { status = 200, body } = await handler(url, init);
    return new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  });
  const calls = (key: string) =>
    fetchImpl.mock.calls
      .map(([input, init]) => ({ url: new URL(String(input), "http://localhost"), init }))
      .filter(
        ({ url, init }) =>
          `${init?.method ?? "GET"} ${url.pathname.replace(/^\/api\/v1/, "")}` === key,
      );
  return { client: createClient(fetchImpl), fetchImpl, calls };
}

export function page<T>(items: T[], total = items.length, limit = 50, offset = 0) {
  return { items, total, limit, offset };
}

const STAMP = "2026-09-14T10:00:00Z";

export function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: 1,
    title: "Backend Engineer",
    department: "Engineering",
    location: "Toronto, ON",
    employment_type: "full_time",
    seniority: "senior",
    education_required: "bachelor",
    min_years_experience: 5,
    description: "Build and run the services behind the hiring pipeline.",
    required_skills: ["Python", "PostgreSQL"],
    preferred_skills: ["Docker"],
    salary_min: "120000.00",
    salary_max: "150000.00",
    status: "open",
    created_at: STAMP,
    updated_at: STAMP,
    requirement_notes: [],
    ...overrides,
  };
}

export function makeCandidate(overrides: Partial<Candidate> = {}): Candidate {
  return {
    id: 10,
    full_name: "Amara Okafor",
    email: "amara.okafor@example.com",
    phone: null,
    location: "Ottawa, ON",
    current_title: "Software Engineer",
    years_experience: 6,
    seniority: "senior",
    education_level: "master",
    resume_text: "Six years building Python services on PostgreSQL.",
    skills: ["Python", "PostgreSQL", "Kubernetes"],
    created_at: STAMP,
    updated_at: STAMP,
    ...overrides,
  };
}

export function makeRanked(
  rank: number,
  candidate: Partial<Candidate>,
  score: number,
  requiredCoverage = 0.5,
): RankedCandidate {
  const full = makeCandidate(candidate);
  return {
    rank,
    score,
    scored_at: STAMP,
    candidate: {
      id: full.id,
      full_name: full.full_name,
      email: full.email,
      current_title: full.current_title ?? null,
      years_experience: full.years_experience,
      seniority: full.seniority,
      skills: full.skills ?? [],
    },
    features: {
      required_skill_coverage: requiredCoverage,
      preferred_skill_coverage: 0,
      skill_jaccard: 0.25,
      experience_gap: 0,
      experience_surplus: 0.1,
      seniority_distance: 0,
      seniority_direction: 0,
      education_fit: 1,
    },
  };
}

export function makeApplication(overrides: Partial<Application> = {}): Application {
  return {
    id: 100,
    job_id: 1,
    candidate_id: 10,
    status: "interview",
    source: "referral",
    notes: null,
    applied_at: STAMP,
    created_at: STAMP,
    updated_at: STAMP,
    ...overrides,
  };
}
