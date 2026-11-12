/**
 * Typed client for the RecruitFlow API.
 *
 * Request and response types come from `schema.ts`, generated from the API's
 * own OpenAPI schema (`npm run gen:api`), so a field renamed on the server is
 * a type error here rather than an `undefined` on the page.
 */
import type { components, paths } from "./schema";

type Schemas = components["schemas"];

export type Health = Schemas["HealthResponse"];
export type Job = Schemas["JobRead"];
export type Candidate = Schemas["CandidateRead"];
export type CandidateSummary = Schemas["CandidateSummary"];
export type Application = Schemas["ApplicationRead"];
export type RankedCandidate = Schemas["RankedCandidate"];
export type MatchFeatures = Schemas["MatchFeatures"];
export type MatchPage = Schemas["MatchPage"];
// Defaulted fields are required in the generated types, which suits responses;
// a request may leave them out and take the server's default.
export type ScoreJobRequest = Partial<Schemas["ScoreJobRequest"]>;
export type ScoreJobResponse = Schemas["ScoreJobResponse"];
export type ApplicationStatus = Schemas["ApplicationStatus"];
export type ResumePreview = Schemas["ResumePreview"];
// The same for the create bodies: what the API requires, and the rest left to
// its defaults.
export type NewJob = Partial<Schemas["JobCreate"]> &
  Pick<Schemas["JobCreate"], "title" | "description">;
export type NewCandidate = Partial<Schemas["CandidateCreate"]> &
  Pick<Schemas["CandidateCreate"], "full_name" | "email" | "resume_text">;
export type NewApplication = Partial<Schemas["ApplicationCreate"]> &
  Pick<Schemas["ApplicationCreate"], "job_id" | "candidate_id">;
export type ApplicationChange = Schemas["ApplicationUpdate"];
// A change names only the fields it sets; a null clears one that may be empty.
export type JobChange = Schemas["JobUpdate"];
export type CandidateChange = Schemas["CandidateUpdate"];
export type Page<T> = { items: T[]; total: number; limit: number; offset: number };

type QueryOf<P extends keyof paths> = paths[P] extends {
  get: { parameters: { query?: infer Q } };
}
  ? NonNullable<Q>
  : never;

export type JobQuery = QueryOf<"/api/v1/jobs">;
export type CandidateQuery = QueryOf<"/api/v1/candidates">;
export type ApplicationQuery = QueryOf<"/api/v1/applications">;
export type MatchQuery = QueryOf<"/api/v1/jobs/{job_id}/matches">;

/** Where the API is mounted. The dev server proxies it; a build is served beside it. */
export const API_BASE = "/api/v1";

/**
 * A request the API refused or could not be sent.
 *
 * `status` is the HTTP status, or 0 when no response arrived at all. `code` is
 * the API's machine-readable error code (`db_unavailable`,
 * `matcher_unavailable`, ...) when the body carried one; `message` is the
 * API's own `detail`, or a summary of a 422's field errors.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string | null;

  constructor(status: number, message: string, code: string | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

type FieldError = { loc?: unknown[]; msg?: string };

function describeError(status: number, body: unknown): { message: string; code: string | null } {
  if (body && typeof body === "object" && "detail" in body) {
    const { detail, code } = body as { detail: unknown; code?: unknown };
    const errorCode = typeof code === "string" ? code : null;
    if (typeof detail === "string") {
      return { message: detail, code: errorCode };
    }
    // A 422 lists each rejected field with where it was found.
    if (Array.isArray(detail)) {
      const parts = (detail as FieldError[]).map((error) => {
        const where = (error.loc ?? []).filter((part) => part !== "body").join(".");
        return where ? `${where}: ${error.msg ?? "invalid"}` : (error.msg ?? "invalid");
      });
      return { message: parts.join("; ") || `Request failed (${status})`, code: errorCode };
    }
  }
  // A gateway answering for an API that is down or not started, as the dev
  // proxy does, sends no body of the API's own.
  if (status >= 502 && status <= 504) {
    return { message: `The API could not be reached (${status})`, code: null };
  }
  return { message: `Request failed (${status})`, code: null };
}

export function buildQuery(params?: object): string {
  if (!params) return "";
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    // A list is the API's repeatable parameter: ?id=1&id=2.
    for (const item of Array.isArray(value) ? value : [value]) {
      if (item !== undefined && item !== null && item !== "") {
        search.append(key, String(item));
      }
    }
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}

type Fetch = typeof fetch;

export type RequestOptions = { signal?: AbortSignal };

/** The client's methods, bound to one `fetch`; tests pass their own. */
export function createClient(fetchImpl: Fetch = (...args) => fetch(...args), base = API_BASE) {
  async function request<T>(
    method: string,
    path: string,
    { signal, body }: RequestOptions & { body?: unknown } = {},
  ): Promise<T> {
    // A file goes as multipart form data, whose Content-Type (with its
    // boundary) the browser writes; anything else goes as JSON.
    const form = body instanceof FormData;
    let response: Response;
    try {
      response = await fetchImpl(`${base}${path}`, {
        method,
        signal,
        headers:
          body === undefined || form
            ? { Accept: "application/json" }
            : { Accept: "application/json", "Content-Type": "application/json" },
        body: body === undefined ? undefined : form ? body : JSON.stringify(body),
      });
    } catch (error) {
      // An abort is the caller's own doing; pass it through untouched.
      if (error instanceof DOMException && error.name === "AbortError") throw error;
      throw new ApiError(0, "The API could not be reached", "network_error");
    }

    const text = await response.text();
    let parsed: unknown = undefined;
    if (text) {
      try {
        parsed = JSON.parse(text);
      } catch {
        parsed = undefined;
      }
    }
    if (!response.ok) {
      const { message, code } = describeError(response.status, parsed);
      throw new ApiError(response.status, message, code);
    }
    if (parsed === undefined) {
      throw new ApiError(response.status, "The API answered with something other than JSON");
    }
    return parsed as T;
  }

  const get = <T>(path: string, options?: RequestOptions) => request<T>("GET", path, options);
  const send = <T>(method: string, path: string, body: unknown, options?: RequestOptions) =>
    request<T>(method, path, { ...options, body });
  const file = (resume: File) => {
    const form = new FormData();
    form.append("file", resume);
    return form;
  };

  return {
    health: (options?: RequestOptions) => get<Health>("/health", options),

    listJobs: (query?: JobQuery, options?: RequestOptions) =>
      get<Page<Job>>(`/jobs${buildQuery(query)}`, options),
    getJob: (jobId: number, options?: RequestOptions) => get<Job>(`/jobs/${jobId}`, options),
    createJob: (job: NewJob, options?: RequestOptions) => send<Job>("POST", "/jobs", job, options),
    updateJob: (jobId: number, change: JobChange, options?: RequestOptions) =>
      send<Job>("PATCH", `/jobs/${jobId}`, change, options),
    /** Closing takes no body. A job already closed is answered as it stands. */
    closeJob: (jobId: number, options?: RequestOptions) =>
      request<Job>("POST", `/jobs/${jobId}/close`, options),

    listCandidates: (query?: CandidateQuery, options?: RequestOptions) =>
      get<Page<Candidate>>(`/candidates${buildQuery(query)}`, options),
    getCandidate: (candidateId: number, options?: RequestOptions) =>
      get<Candidate>(`/candidates/${candidateId}`, options),
    createCandidate: (candidate: NewCandidate, options?: RequestOptions) =>
      send<Candidate>("POST", "/candidates", candidate, options),
    updateCandidate: (candidateId: number, change: CandidateChange, options?: RequestOptions) =>
      send<Candidate>("PATCH", `/candidates/${candidateId}`, change, options),
    /** The text and skills the API reads from a resume file, storing nothing. */
    previewResume: (resume: File, options?: RequestOptions) =>
      send<ResumePreview>("POST", "/candidates/resume-preview", file(resume), options),
    uploadResume: (candidateId: number, resume: File, options?: RequestOptions) =>
      send<Candidate>("PUT", `/candidates/${candidateId}/resume`, file(resume), options),

    listApplications: (query?: ApplicationQuery, options?: RequestOptions) =>
      get<Page<Application>>(`/applications${buildQuery(query)}`, options),
    createApplication: (application: NewApplication, options?: RequestOptions) =>
      send<Application>("POST", "/applications", application, options),
    updateApplication: (applicationId: number, change: ApplicationChange, options?: RequestOptions) =>
      send<Application>("PATCH", `/applications/${applicationId}`, change, options),
    reopenApplication: (applicationId: number, options?: RequestOptions) =>
      send<Application>("POST", `/applications/${applicationId}/reopen`, {}, options),

    listMatches: (jobId: number, query?: MatchQuery, options?: RequestOptions) =>
      get<MatchPage>(`/jobs/${jobId}/matches${buildQuery(query)}`, options),
    scoreJob: (jobId: number, body: ScoreJobRequest = {}, options?: RequestOptions) =>
      request<ScoreJobResponse>("POST", `/jobs/${jobId}/score`, { ...options, body }),

    /** Links rather than calls: the browser downloads the PDFs itself. */
    shortlistPdfUrl: (jobId: number, limit?: number) =>
      `${base}/jobs/${jobId}/report.pdf${buildQuery({ limit })}`,
    scorecardPdfUrl: (jobId: number, candidateId: number) =>
      `${base}/jobs/${jobId}/candidates/${candidateId}/scorecard.pdf`,
  };
}

export type ApiClient = ReturnType<typeof createClient>;

export const api = createClient();
